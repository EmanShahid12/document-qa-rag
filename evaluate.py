import json
import sys
from rag import DocumentQA


def main(path: str = "eval_set.json"):
    with open(path, encoding="utf-8-sig") as f:
        items = json.load(f)
    qa = DocumentQA()
    qa.load_index()
    ans_total = retrieval_hits = keyword_ok = 0
    unans_total = abstained = 0
    rows = []

    for item in items:
        question = item["question"]
        answerable = item.get("answerable", True)
        retrieved_files = {doc.metadata.get("file") for doc, _ in qa.retrieve(question)}
        result = qa.ask(question)

        row = {
            "question": question,
            "answerable": answerable,
            "status": result.status,
            "confidence": result.confidence,
            "answer": result.answer,
        }

        if answerable:
            ans_total += 1
            expected_file = item.get("expected_file")
            if expected_file:
                hit = expected_file in retrieved_files
                retrieval_hits += int(hit)
                row["retrieval_hit"] = hit
            keywords = [k.lower() for k in item.get("expected_keywords", [])]
            if keywords:
                ok = all(k in result.answer.lower() for k in keywords)
                keyword_ok += int(ok)
                row["keywords_ok"] = ok
        else:
            unans_total += 1
            ok = result.status == "not_found"
            abstained += int(ok)
            row["correctly_abstained"] = ok

        rows.append(row)

    print(f"Answerable questions      : {ans_total}")
    print(f"  retrieval hit (file)    : {retrieval_hits}/{ans_total}")
    print(f"  keyword match           : {keyword_ok}/{ans_total}")
    print(f"Unanswerable questions    : {unans_total}")
    print(f"  correctly said not found: {abstained}/{unans_total}")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("Details saved to eval_results.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_set.json")