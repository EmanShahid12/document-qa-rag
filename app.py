import argparse
from rag import DocumentQA, RAGAnswer
STATUS_LABEL = {
    "grounded": "GROUNDED",
    "low_confidence": "LOW CONFIDENCE - verify with the sources",
    "not_found": "NOT FOUND IN DOCUMENTS",
    "web_fallback": "WEB FALLBACK (not from your documents)",
}
def print_answer(r: RAGAnswer):
    print(f"\n[{STATUS_LABEL[r.status]}]  retrieval_score={r.confidence}")
    print(f"\n{r.answer}\n")
    if r.sources:
        print("Sources:")
        for i, s in enumerate(r.sources, start=1):
            where = f"{s.file}" + (f", page {s.page}" if s.page else "")
            print(f"  - {where} (chunk {s.chunk_id}, score {s.score})")
            print(f"    \"{s.snippet}...\"")


def main():
    parser = argparse.ArgumentParser(description="Document Q&A (RAG with citations)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="Build the FAISS index from a folder")
    p.add_argument("--data", default="data")

    p = sub.add_parser("ask", help="Ask one question")
    p.add_argument("question")
    p.add_argument("--web", action="store_true", help="Use web fallback when docs lack the answer")

    p = sub.add_parser("chat", help="Interactive Q&A loop")
    p.add_argument("--web", action="store_true")

    p = sub.add_parser("summarize", help="Summarize one indexed file")
    p.add_argument("file")

    p = sub.add_parser("quiz", help="Generate a multiple-choice quiz on a topic")
    p.add_argument("topic")
    p.add_argument("-n", type=int, default=5)

    args = parser.parse_args()
    qa = DocumentQA()

    if args.cmd == "index":
        n_docs, n_chunks = qa.build_index(args.data)
        print(f"Indexed {n_docs} pages/documents into {n_chunks} chunks.")
        return

    qa.load_index()

    if args.cmd == "ask":
        print_answer(qa.ask(args.question, web_fallback=args.web))
    elif args.cmd == "chat":
        print("Type 'exit' to quit.")
        while True:
            q = input("\nQuestion> ").strip()
            if q.lower() in {"exit", "quit", ""}:
                break
            print_answer(qa.ask(q, web_fallback=args.web))
    elif args.cmd == "summarize":
        print(qa.summarize(args.file))
    elif args.cmd == "quiz":
        quiz = qa.quiz(args.topic, args.n)
        for i, q in enumerate(quiz.questions, start=1):
            print(f"\nQ{i}. {q.question}")
            for letter, opt in zip("ABCD", q.options):
                print(f"   {letter}. {opt}")
            print(f"   Answer: {'ABCD'[q.answer_index]} - {q.explanation}")


if __name__ == "__main__":
    main()