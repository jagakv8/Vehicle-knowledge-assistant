from assistant import (
    SentenceTransformer,
    load_documents,
    build_search_index,
    retrieve_documents,
    ABSTENTION_TEXT,
    MIN_SIMILARITY,
)


TESTS = [
    {
        "question": "What happens if the fuel pump fails?",
        "expected_campaigns": {"21V215000", "23V858000"},
    },
    {
        "question": "What is the remedy for the body control module issue?",
        "expected_campaigns": {"20V771000"},
    },
    {
        "question": "Which recalls involve airbag sensors?",
        "expected_campaigns": {"23V865000", "24V064000", "26V332000"},
    },
    {
        "question": (
            "What is the recall campaign for a 2024 Ferrari with a transmission defect?"
        ),
        "expected_campaigns": set(),
    },
]


def main():
    print("Loading evaluation dataset...")
    documents = load_documents()

    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Building index...")
    index = build_search_index(documents, model)

    passed = 0

    for test in TESTS:
        question = test["question"]
        expected = test["expected_campaigns"]

        results = retrieve_documents(question, documents, model, index)

        found = {result["campaign"] for result in results if result["campaign"]}

        if expected:
            success = bool(found.intersection(expected))
        else:
            success = len(results) == 0

        print("\n" + "-" * 60)
        print("Question:", question)
        print("Expected campaigns:", sorted(expected) or "Abstain")
        print("Retrieved campaigns:", sorted(found) or "None")
        print("Result:", "PASS" if success else "FAIL")

        if success:
            passed += 1

    print("\n" + "=" * 60)
    print(f"Evaluation passed: {passed}/{len(TESTS)}")
    print(f"Configured similarity threshold: {MIN_SIMILARITY}")
    print(
        "Note: this measures retrieval behavior, not full answer "
        "correctness or citation faithfulness."
    )


if __name__ == "__main__":
    main()
