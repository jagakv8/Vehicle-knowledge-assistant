import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data", "recalls.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "recall_documents.json")


def prepare_recall_documents():
    if not os.path.exists(INPUT_FILE):
        print("Error: data/recalls.json was not found.")
        print("Run main.py and search for a vehicle first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        recalls = json.load(file)

    documents = []

    for recall in recalls:
        vehicle = recall.get("_searched_vehicle", {})

        text = "\n".join(
            [
                f"Campaign: {recall.get('NHTSACampaignNumber', 'Unknown')}",
                f"Component: {recall.get('Component', 'Unknown')}",
                f"Summary: {recall.get('Summary', 'Unknown')}",
                f"Consequence: {recall.get('Consequence', 'Unknown')}",
                f"Remedy: {recall.get('Remedy', 'Unknown')}",
            ]
        )

        documents.append(
            {
                "id": recall.get("NHTSACampaignNumber", "unknown"),
                "vehicle": vehicle,
                "text": text,
                "source": "NHTSA recall API",
            }
        )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(documents, file, indent=2, ensure_ascii=False)

    print(f"Prepared {len(documents)} recall documents.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    prepare_recall_documents()
