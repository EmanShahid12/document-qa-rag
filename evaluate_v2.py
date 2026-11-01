import argparse
import json
import re
from collections import defaultdict
from rag import DocumentQA

NUMERIC_CITATION = re.compile(r"\[(\d+)\]")
def frac(n, m):
    return f"{n}/{m}"
def run_once(qa, items):
    rows = []
    for item in items:
        q = item["question"]
        answerable = item.get("answerable", True)
        hits = qa.retrieve(q)  
        result = qa.ask(q)

        pages = [doc.metadata.get("page") for doc, _ in hits]
        pages = [p + 1 if p is not None else None for p in pages]  
        row = {
            "question": q,
            "answerable": answerable,
            "kind": item.get("kind") or item.get("topic"),
            "status": result.status,
            "confidence": result.confidence,
            "retrieved_pages": pages,
            "answer": result.answer,
        }

        if answerable:
            expected = set(item.get("expected_pages", []))
            if expected:
                row["page_hit"] = any(p in expected for p in pages)
                row["top1_page_hit"] = bool(pages) and pages[0] in expected
                cited_pages = {s.page for s in result.sources}
                row["cited_page_hit"] = bool(cited_pages & expected)
            kws = [k.lower() for k in item.get("expected_keywords", [])]
            if kws:
                row["keywords_ok"] = all(k in result.answer.lower() for k in kws)
            row["has_numeric_citation"] = bool(NUMERIC_CITATION.search(result.answer))
            row["answered"] = result.status != "not_found"
            row["manual_correct"] = None  # <- fill in by hand after reading the answer
        else:
            row["correctly_abstained"] = result.status == "not_found"
            row["manual_correct"] = None
        rows.append(row)
    return rows


def summarize(rows):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    print(f"\nAnswerable questions: {len(ans)}")
    for key, label in [
        ("page_hit", "retrieval page hit (any of top-k)"),
        ("top1_page_hit", "retrieval page hit (rank 1)"),
        ("cited_page_hit", "cited source page is correct"),
        ("keywords_ok", "keyword match (crude)"),
        ("has_numeric_citation", "answer has a valid-style [n] citation"),
        ("answered", "answered (did not say not_found)"),
    ]:
        scored = [r for r in ans if key in r]
        print(f"  {label:<42}: {frac(sum(bool(r[key]) for r in scored), len(scored))}")
    by_status = defaultdict(int)
    for r in ans:
        by_status[r["status"]] += 1
    print("  status breakdown:", dict(by_status))

    print(f"\nUnanswerable questions: {len(unans)}")
    print(f"  correctly said not_found: {frac(sum(r['correctly_abstained'] for r in unans), len(unans))}")
    kinds = defaultdict(lambda: [0, 0])
    for r in unans:
        kinds[r["kind"]][1] += 1
        kinds[r["kind"]][0] += int(r["correctly_abstained"])
    for k, (n, m) in kinds.items():
        print(f"    {k:<30}: {frac(n, m)}")

    print("\nPlease read these manually:")
    for r in ans:
        problems = []
        if r.get("page_hit") is False:
            problems.append("retrieval missed expected page")
        if r.get("keywords_ok") is False:
            problems.append("keywords missing")
        if r["status"] == "not_found":
            problems.append("said not_found")
        if r["status"] == "low_confidence":
            problems.append("low_confidence")
        if not r["has_numeric_citation"] and r["status"] != "not_found":
            problems.append("no [n] citation")
        if problems:
            print(f"  - {r['question']}\n      -> {', '.join(problems)}")
    for r in unans:
        if not r["correctly_abstained"]:
            print(f"  - WRONGLY ANSWERED unanswerable: {r['question']}\n      -> {r['answer'][:150]!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="eval_set_v2.json")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", default="eval_results_v2.json")
    args = parser.parse_args()

    with open(args.path, encoding="utf-8-sig") as f:
        items = json.load(f)

    qa = DocumentQA()
    qa.load_index()

    all_runs = []
    for run in range(1, args.runs + 1):
        print(f"\n===== RUN {run}/{args.runs} =====")
        rows = run_once(qa, items)
        summarize(rows)
        all_runs.append(rows)

    if args.runs > 1:
        flaky = []
        for i, item in enumerate(items):
            statuses = {run[i]["status"] for run in all_runs}
            if len(statuses) > 1:
                flaky.append((item["question"], sorted(statuses)))
        print(f"\nQuestions whose status changed between runs: {len(flaky)}/{len(items)}")
        for q, s in flaky:
            print(f"  - {q} -> {s}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_runs[0] if args.runs == 1 else all_runs, f, indent=2, ensure_ascii=False)
    print(f"\nDetails saved to {args.output}")


if __name__ == "__main__":
    main()
