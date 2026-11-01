from __future__ import annotations
import json
from typing import List
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError
from rag import DocumentQA, Quiz


def build_tools(qa: DocumentQA) -> List[BaseTool]:
    @tool
    def document_search(question: str) -> str:
        """Answer a question from indexed documents, including citations."""
        return qa.ask(question, web_fallback=False).model_dump_json(indent=2)

    @tool
    def summarize_document(file_name: str) -> str:
        """Summarize a document by exact indexed filename, such as data.pdf."""
        return qa.summarize(file_name)

    @tool
    def generate_quiz(topic: str, n_questions: int = 5) -> str:
        """Generate 1 to 10 multiple-choice questions grounded in indexed documents."""
        try:
            n_questions = int(n_questions)
        except (TypeError, ValueError):
            return "ERROR: n_questions must be an integer."
        if not 1 <= n_questions <= 10:
            return "ERROR: n_questions must be between 1 and 10."

        try:
            quiz = qa.quiz(topic, n_questions)
            if len(quiz.questions) == n_questions:
                return quiz.model_dump_json(indent=2)
        except (RuntimeError, ValidationError) as original_error:
            pass  # Try model-enforced structured output below.
        except Exception as exc:
            return f"ERROR: Quiz generation failed: {type(exc).__name__}: {exc}"

        try:
            hits = qa.retrieve(topic, k=6)
            if not hits:
                return "ERROR: No supporting document passages were found."
            context = "\n\n".join(doc.page_content for doc, _ in hits)
            structured_llm = qa.llm.with_structured_output(Quiz, method="json_schema")
            prompt = (
                f"Create exactly {n_questions} distinct multiple-choice questions about '{topic}' "
                "using ONLY the provided passages. Each question must have exactly four "
                "options, answer_index must be 0, 1, 2 or 3, and each explanation must "
                "be supported by the passages. Do not use outside knowledge.\n\n"
                f"Passages:\n{context}"
            )
            quiz = structured_llm.invoke(prompt)
            if isinstance(quiz, dict):
                quiz = Quiz.model_validate(quiz)
            if not isinstance(quiz, Quiz) or len(quiz.questions) != n_questions:
                return "ERROR: Quiz output failed schema or question-count validation."
            return quiz.model_dump_json(indent=2)
        except Exception as exc:
            return f"ERROR: Structured quiz generation failed: {type(exc).__name__}: {exc}"

    @tool
    def list_documents() -> str:
        """List filenames available in the indexed document collection."""
        files = sorted({str(c.metadata.get("file", "unknown")) for c in qa._all_chunks()})
        return json.dumps({"files": files}, ensure_ascii=False)

    return [document_search, summarize_document, generate_quiz, list_documents]
