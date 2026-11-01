"""
Find out WHY a question fails: is the answer chunk missing, not retrieved, or mis-scored?

    python debug_find.py "six chapters" "How many chapters does the book consist of?"

Arg 1: a phrase you KNOW appears in the answer passage of your document.
Arg 2: the question you are asking.
"""
import sys

from rag import CANDIDATES, DocumentQA

phrase = sys.argv[1] if len(sys.argv) > 1 else "six chapters"
question = sys.argv[2] if len(sys.argv) > 2 else "How many chapters does the book consist of?"

qa = DocumentQA()
qa.load_index()
all_chunks = qa._all_chunks()

matches = [c for c in all_chunks if phrase.lower() in c.page_content.lower()]
print(f"\nTotal chunks in index: {len(all_chunks)}")
print(f"Chunks containing '{phrase}': {len(matches)}")
if not matches:
    print("-> The phrase is not inside any single chunk (different wording, or split across two chunks).")
    print("   Try a shorter phrase, e.g. 'chapters'.")
    raise SystemExit

ranked = qa.db.similarity_search_with_score(question, k=len(all_chunks))
faiss_rank = {doc.metadata["chunk_id"]: i + 1 for i, (doc, _) in enumerate(ranked)}

for chunk in matches[:5]:
    cid = chunk.metadata["chunk_id"]
    logit = float(qa.reranker.predict([(question, chunk.page_content)])[0])
    rank = faiss_rank[cid]
    verdict = "reached the reranker" if rank <= CANDIDATES else f"NEVER reached the reranker (only top {CANDIDATES} go through)"
    print(f"\nchunk {cid} | page {chunk.metadata.get('page')} | FAISS rank {rank} -> {verdict}")
    print(f"reranker logit: {logit:.2f}  (above 0 = reranker thinks it answers the question)")
    print(chunk.page_content[:300].replace("\n", " "))