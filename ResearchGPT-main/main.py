import os
from research_assistant import ResearchGPTAssistant
from document_processor import DocumentProcessor
import config


def main():
    print("\n=== ResearchGPT Assistant ===\n")

    print("Initializing document processor...")
    doc_processor = DocumentProcessor(data_dir=os.path.join(os.getcwd(), "data"))
    doc_processor.build_search_index()
    stats = doc_processor.get_document_stats()
    print(f"Loaded {stats['num_docs']} document(s) with {stats['num_chunks']} chunks.\n")

    print("Connecting to Mistral API...")
    assistant = ResearchGPTAssistant(config, doc_processor)
    if not assistant.mistral_client:
        print("Warning: Mistral client not initialized. Check your API key in config.py.\n")

    print("Type your research question below. Type 'exit' or 'quit' to end.\n")
    while True:
        query = input(" Ask a question: ").strip()
        if query.lower() in ("exit", "quit"):
            print("\n Exiting ResearchGPT Assistant. Goodbye!")
            break

        if not query:
            continue

        print("\n Thinking...\n")
        result = assistant.answer_research_question(query)
        print(f"\n Answer:\n{result['answer']}\n")

        if result.get("verification"):
            print(" Verified / Improved Answer:")
            print(result["verification"]["improved_answer"])
            print("\n")

        print(" Sources used:", result.get("sources_used", []))
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
