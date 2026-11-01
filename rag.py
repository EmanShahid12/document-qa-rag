"""
Document Q&A with citation grounding.

Stack: LangChain + FAISS + HuggingFace sentence-transformers + Ollama (local LLM).

Features
- PDF / TXT / MD ingestion with file + page + chunk metadata
- Answers with numbered citations ([1], [2]) mapped back to real sources
- Confidence gating: grounded / low_confidence / not_found
- Optional web fallback (DuckDuckGo) when documents do not contain the answer
- Pydantic-validated structured output (RAGAnswer, Quiz)
- Document summarization (map-reduce) and quiz generation (validated JSON, with retries)
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import List, Literal, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, ValidationError
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ----------------------------- configuration -----------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # any model you have pulled in Ollama
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 4            # chunks given to the LLM
CANDIDATES = 150     # chunks fetched from FAISS (semantic) before reranking
BM25_CANDIDATES = 30  # chunks fetched by BM25 (exact keywords) before reranking

# Cross-encoder reranker: re-scores (question, chunk) pairs far more accurately
# than embedding similarity alone. Downloaded from Hugging Face on first use.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Scores are reranker probabilities in [0, 1]. These are STARTING values, not tested
# results: tune them on your own documents using evaluate.py.
HIGH_CONF = float(os.getenv("HIGH_CONF", "0.5"))
LOW_CONF = float(os.getenv("LOW_CONF", "0.05"))

NOT_FOUND_MESSAGE = "I could not find the answer to this question in the provided documents."


# ----------------------------- data models -------------------------------
class Source(BaseModel):
    file: str
    page: Optional[int] = None
    chunk_id: Optional[int] = None
    score: float = Field(ge=0, le=1)
    snippet: str


class RAGAnswer(BaseModel):
    question: str
    answer: str
    status: Literal["grounded", "low_confidence", "not_found", "web_fallback"]
    # Retrieval confidence of the best document chunk (0 for web fallback).
    confidence: float = Field(ge=0, le=1)
    sources: List[Source]


class QuizQuestion(BaseModel):
    question: str
    options: List[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    explanation: str


class Quiz(BaseModel):
    topic: str
    questions: List[QuizQuestion]


# ----------------------------- helpers -----------------------------------
def load_documents(data_dir: str):
    docs = []
    for path in sorted(Path(data_dir).rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        elif suffix in {".txt", ".md"}:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        else:
            continue
        for d in loaded:
            d.metadata["file"] = path.name
        docs.extend(loaded)
    if not docs:
        raise FileNotFoundError(f"No .pdf/.txt/.md files found in '{data_dir}'.")
    return docs


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def _is_junk(text: str) -> bool:
    """Drop table-of-contents dot leaders, near-empty chunks and mostly-symbol chunks."""
    if re.search(r"(\. ){5,}", text):
        return True
    stripped = text.strip()
    if len(stripped) < 80:
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    return letters < 0.5 * len(stripped)


def _page_of(doc) -> Optional[int]:
    page = doc.metadata.get("page")
    return int(page) + 1 if page is not None else None  # PyPDF pages are 0-indexed


def _to_source(doc, score: float) -> Source:
    snippet = doc.page_content.strip().replace("\n", " ")
    return Source(
        file=doc.metadata.get("file", "unknown"),
        page=_page_of(doc),
        chunk_id=doc.metadata.get("chunk_id"),
        score=round(score, 4),
        snippet=snippet[:240],
    )


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


# ----------------------------- main class --------------------------------
class DocumentQA:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.db: Optional[FAISS] = None
        self._chunks = []
        self._bm25 = None

    # ---- indexing ----
    def build_index(self, data_dir: str):
        docs = load_documents(data_dir)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = [c for c in splitter.split_documents(docs) if not _is_junk(c.page_content)]
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        self.db = FAISS.from_documents(chunks, self.embeddings)
        self.db.save_local(self.index_dir)
        self._build_bm25()
        return len(docs), len(chunks)

    def load_index(self):
        self.db = FAISS.load_local(
            self.index_dir, self.embeddings, allow_dangerous_deserialization=True
        )
        self._build_bm25()

    # ---- retrieval ----
    def _build_bm25(self):
        self._chunks = self._all_chunks()
        self._bm25 = BM25Okapi([_tokenize(c.page_content) for c in self._chunks])

    def retrieve(self, question: str, k: int = TOP_K):
        """Hybrid retrieval: FAISS (semantic) + BM25 (keywords) -> cross-encoder rerank -> top k."""
        faiss_docs = [d for d, _ in self.db.similarity_search_with_score(question, k=CANDIDATES)]

        bm25_scores = self._bm25.get_scores(_tokenize(question))
        top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_docs = [self._chunks[i] for i in top_idx[:BM25_CANDIDATES] if bm25_scores[i] > 0]

        seen, candidates = set(), []
        for doc in faiss_docs + bm25_docs:
            cid = doc.metadata.get("chunk_id")
            if cid not in seen:
                seen.add(cid)
                candidates.append(doc)
        if not candidates:
            return []

        logits = self.reranker.predict([(question, doc.page_content) for doc in candidates])
        scored = [
            (doc, 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(logit))))))
            for doc, logit in zip(candidates, logits)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def _all_chunks(self):
        chunks = list(self.db.docstore._dict.values())
        return sorted(chunks, key=lambda d: d.metadata.get("chunk_id", 0))

    # ---- question answering ----
    def ask(self, question: str, k: int = TOP_K, web_fallback: bool = False) -> RAGAnswer:
        hits = self.retrieve(question, k)
        top_score = hits[0][1] if hits else 0.0

        if top_score < LOW_CONF:
            return self._not_found(question, top_score, web_fallback)

        context = "\n\n".join(
            f"[{i}] ({doc.metadata.get('file')}, page {_page_of(doc)})\n{doc.page_content}"
            for i, (doc, _) in enumerate(hits, start=1)
        )
        prompt = (
            "You are a document question-answering assistant.\n"
    "Answer the user's question using ONLY the context passages below.\n"
    "The passages may describe the answer using different wording.\n"
    "Combine relevant information from multiple passages when useful.\n"
    "If the context contains partial but useful information, provide that "
"information and clearly state its scope.\n"
"If the context does not contain enough evidence to answer the specific "
"question, reply exactly NOT_FOUND.\n"
"Never guess or use outside knowledge to answer.\n"
    "Every answer must include citations like [1] or [2]. "
"Put a citation immediately after each factual claim. "
"Use only passage numbers that exist in the context.\n"
    "Do not invent facts or citations.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {question}\n"
    "Answer:"
        )
        text = self.llm.invoke(prompt).content.strip()
        #print("DEBUG - Ollama response:", repr(text))
        #print("\nDEBUG - Retrieved context:")
        #print(context[:3000])
        if os.getenv("RAG_DEBUG"):
            print(f"DEBUG - Ollama response: {text!r}")

        normalized_text = re.sub(r"[^A-Za-z]", "", text).upper()

        if normalized_text == "NOTFOUND":
             return self._not_found(question, top_score, web_fallback)

        valid = set(range(1, len(hits) + 1))
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)} & valid)
        if not cited:
            # Model gave no usable [n] citation (e.g. "[Manning, 2022]"), so ask once more.
            retry = self.llm.invoke(
                prompt
                + "\n\nIMPORTANT: your previous answer had no numeric citation. "
                "Answer again and put a numeric citation such as [1] after every factual claim. "
                "If the context cannot answer the question, reply exactly NOT_FOUND."
            ).content.strip()
            if re.sub(r"[^A-Za-z]", "", retry).upper() == "NOTFOUND":
                return self._not_found(question, top_score, web_fallback)
            retry_cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", retry)} & valid)
            if retry_cited:
                text, cited = retry, retry_cited
        used = [hits[i - 1] for i in cited] if cited else hits
        status = "grounded" if (cited and top_score >= HIGH_CONF) else "low_confidence"

        return RAGAnswer(
            question=question,
            answer=text,
            status=status,
            confidence=round(top_score, 4),
            sources=[_to_source(doc, score) for doc, score in used],
        )

    def _not_found(self, question: str, top_score: float, web_fallback: bool) -> RAGAnswer:
        if web_fallback:
            return self._web_answer(question)
        return RAGAnswer(
            question=question,
            answer=NOT_FOUND_MESSAGE,
            status="not_found",
            confidence=round(top_score, 4),
            sources=[],
        )

    def _web_answer(self, question: str) -> RAGAnswer:
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise RuntimeError("Web fallback needs: pip install duckduckgo-search") from e

        with DDGS() as ddgs:
            results = list(ddgs.text(question, max_results=4))
        if not results:
            return RAGAnswer(
                question=question, answer=NOT_FOUND_MESSAGE,
                status="not_found", confidence=0.0, sources=[],
            )

        context = "\n\n".join(
            f"[{i}] ({r.get('href')})\n{r.get('body', '')}" for i, r in enumerate(results, start=1)
        )
        prompt = (
            "Answer using ONLY the numbered web snippets below and cite them like [1].\n"
            "If they do not answer the question, reply exactly: NOT_FOUND\n\n"
            f"Snippets:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        text = self.llm.invoke(prompt).content.strip()
        if "NOT_FOUND" in text:
            return RAGAnswer(
                question=question, answer=NOT_FOUND_MESSAGE,
                status="not_found", confidence=0.0, sources=[],
            )
        sources = [
            Source(file=r.get("href", "web"), score=0.0, snippet=(r.get("body") or "")[:240])
            for r in results
        ]
        return RAGAnswer(
            question=question, answer=text, status="web_fallback", confidence=0.0, sources=sources
        )

    # ---- summarization (map-reduce) ----
    def summarize(self, file_name: str, batch_size: int = 6, max_chunks: int = 60) -> str:
        chunks = [c for c in self._all_chunks() if c.metadata.get("file") == file_name][:max_chunks]
        if not chunks:
            raise ValueError(f"'{file_name}' not found in index.")

        partials = []
        for i in range(0, len(chunks), batch_size):
            text = "\n\n".join(c.page_content for c in chunks[i : i + batch_size])
            partials.append(
                self.llm.invoke(
                    f"Summarize the following text in 3-5 concise bullet points:\n\n{text}"
                ).content.strip()
            )
        if len(partials) == 1:
            return partials[0]
        combined = "\n\n".join(partials)
        return self.llm.invoke(
            "Combine these partial summaries into one coherent summary "
            f"(max 8 bullet points):\n\n{combined}"
        ).content.strip()

    # ---- quiz generation (validated JSON) ----
    def quiz(self, topic: str, n_questions: int = 5, retries: int = 3) -> Quiz:
        hits = self.retrieve(topic, k=6)
        if not hits or hits[0][1] < LOW_CONF:
            raise ValueError("Topic not found in the documents.")
        context = "\n\n".join(doc.page_content for doc, _ in hits)
        base_prompt = (
            f"Using ONLY the context below, write {n_questions} multiple-choice questions "
            f"about '{topic}'.\nReturn ONLY valid JSON in exactly this format:\n"
            '{"topic": "...", "questions": [{"question": "...", "options": ["A","B","C","D"], '
            '"answer_index": 0, "explanation": "..."}]}\n'
            "Each question has exactly 4 options and answer_index is 0-3.\n\n"
            f"Context:\n{context}"
        )
        hint = ""
        for _ in range(retries):
            raw = self.llm.invoke(base_prompt + hint).content
            try:
                return Quiz.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as e:
                hint = f"\n\nYour previous output was invalid ({str(e)[:300]}). Return ONLY valid JSON."
        raise RuntimeError("Could not get a valid quiz from the model after several attempts.")