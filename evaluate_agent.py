"""Step 4: offline checks + optional live LangGraph evaluation.
Preserves all existing project files. Run from your project directory.
"""
from __future__ import annotations
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from graph_agent_step3 import requested_tools, quiz_review, render_tool, ReliableGraphAgent


def check(name, ok, detail, results):
    results.append({"test": name, "passed": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def offline_checks(results):
    cases = [
        ("Document routing", "What is pre-training?", {"document_search"}),
        ("Quiz routing", "Generate 5 MCQs about fine-tuning", {"generate_quiz"}),
        ("Summary routing", "Summarize data.pdf", {"summarize_document"}),
        ("Multi-tool routing", "Summarize data.pdf and generate 5 MCQs about pre-training", {"summarize_document", "generate_quiz"}),
        ("List routing", "List available documents", {"list_documents"}),
    ]
    for name, question, expected in cases:
        got = requested_tools(question)
        check(name, got == expected, f"expected={sorted(expected)}, got={sorted(got)}", results)

    sample = json.dumps({"topic": "pre-training", "questions": [{
        "question": "What is pre-training?", "options": ["A", "B", "C", "D"],
        "answer_index": 1, "explanation": "The supplied context supports B."
    }]})
    formatted, warnings = quiz_review(sample, expected_count=1)
    check("Valid quiz schema", bool(formatted) and not warnings, f"warnings={warnings}", results)
    _, warnings = quiz_review(sample, expected_count=5)
    check("Quiz count mismatch detected", any("Expected 5" in w for w in warnings), str(warnings), results)
    invalid = json.dumps({"topic": "x", "questions": [{"question": "x", "options": ["A", "B"], "answer_index": 7}]})
    _, warnings = quiz_review(invalid, expected_count=1)
    check("Invalid quiz answer index detected", any("invalid" in w for w in warnings), str(warnings), results)
    not_found = json.dumps({"status": "not_found", "answer": "I could not find the answer.", "sources": []})
    output = render_tool("document_search", not_found)
    check("Not-found safely rendered", "not_found" in output and "insufficient" in output.lower(), output[:130], results)
    error = render_tool("generate_quiz", "ERROR: RuntimeError: invalid quiz")
    check("Tool error not fabricated", "FAILED" in error and "No answer was invented" in error, error[:140], results)


def live_checks(results, db):
    # Fresh thread avoids contamination from prior chat. SQLite file is shared but isolated by thread_id.
    thread = "eval_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    agent = ReliableGraphAgent(db_path=db)
    try:
        def ask(question):
            return agent.ask(question, thread_id=thread, verbose=False)

        first = ask("What is pre-training in large language models?")
        check("LIVE document retrieval", "status: grounded" in first.lower() and "sources:" in first.lower(), first[:200], results)
        followup = ask("What are its advantages?")
        check("LIVE follow-up context", "status: grounded" in followup.lower() and bool(re.search(r"pre.train", followup, re.I)), followup[:200], results)
        recall = ask("What topic were we discussing earlier?")
        check("LIVE conversation recall", "conversation memory" in recall.lower() and bool(re.search(r"pre.train", recall, re.I)), recall[:200], results)
        unknown = ask("What is the salary of the CEO mentioned in data.pdf?")
        check("LIVE unsupported answer refusal", "not_found" in unknown.lower() or "low_confidence" in unknown.lower(), unknown[:200], results)
        quiz = ask("Generate 5 MCQs about fine-tuning")
        check("LIVE quiz generation", "Quiz:" in quiz and quiz.count("  Answer:") == 5, quiz[:200], results)
        both = ask("Summarize data.pdf and generate 5 MCQs about pre-training")
        check("LIVE multiple tools", "summarize_document:" in both and "Quiz:" in both, both[:200], results)
    finally:
        agent.close()


def main():
    parser = argparse.ArgumentParser(description="Step 4 Agent evaluation")
    parser.add_argument("--live", action="store_true", help="Run real Ollama/RAG integration tests (slower)")
    parser.add_argument("--db", default="agent_memory.sqlite3")
    parser.add_argument("--output", default="evaluation_report.json")
    args = parser.parse_args()
    results = []
    offline_checks(results)
    if args.live:
        try:
            live_checks(results, args.db)
        except Exception as exc:
            check("LIVE suite execution", False, f"{type(exc).__name__}: {exc}", results)
    passed = sum(r["passed"] for r in results)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "live+offline" if args.live else "offline",
        "passed": passed, "total": len(results),
        "pass_rate_percent": round(100 * passed / len(results), 1) if results else 0,
        "caveats": [
            "Offline checks test deterministic routing/formatting only, not model behavior.",
            "Live checks use approximate expected text and can be flaky with local LLMs.",
            "Quiz answer factual accuracy and citation entailment are NOT independently verified.",
            "A pass rate from this small test suite is NOT an overall agent accuracy metric."
        ],
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRESULT: {passed}/{len(results)} checks passed ({report['pass_rate_percent']}%).")
    print(f"Report saved: {Path(args.output).resolve()}")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
