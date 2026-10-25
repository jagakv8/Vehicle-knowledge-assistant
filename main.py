import requests
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "recalls.json")


def lookup_recalls(make, model, year):
    url = "https://api.nhtsa.gov/recalls/recallsByVehicle"

    params = {
        "make": make,
        "model": model,
        "modelYear": year,
    }

    print(f"\nSearching for {year} {make} {model}...")

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as error:
        print(f"API request failed: {error}")
        return

    recalls = data.get("results") or []

    # Create the data folder if needed
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load previous records
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
            previous = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        previous = []

    if not isinstance(previous, list):
        previous = []

    # Add the vehicle details to each record
    for recall in recalls:
        recall["_searched_vehicle"] = {
            "make": make,
            "model": model,
            "year": year,
        }

    # Merge records and avoid duplicates
    combined = {
        item.get("NHTSACampaignNumber"): item
        for item in previous
        if isinstance(item, dict) and item.get("NHTSACampaignNumber")
    }

    for item in recalls:
        campaign = item.get("NHTSACampaignNumber")
        if campaign:
            combined[campaign] = item

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(
                list(combined.values()),
                file,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as error:
        print(f"Could not save records: {error}")
        return

    print(f"Records returned for this search: {len(recalls)}")
    print(f"Unique campaigns saved: {len(combined)}")
    print(f"Saved file: {OUTPUT_FILE}")

    for index, recall in enumerate(recalls, start=1):
        print(f"\n--- Recall {index} ---")
        print("Campaign:", recall.get("NHTSACampaignNumber", "N/A"))
        print("Component:", recall.get("Component", "N/A"))
        print("Summary:", recall.get("Summary", "N/A"))
        print("Remedy:", recall.get("Remedy", "N/A"))

    if not recalls:
        print("No records returned. This does not prove there are no recalls.")


def main():
    print("\n=== Vehicle Knowledge Assistant ===")

    make = input("Vehicle make: ").strip()
    model = input("Vehicle model: ").strip()
    year_text = input("Model year: ").strip()

    if not make or not model:
        print("Make and model cannot be empty.")
        return

    try:
        year = int(year_text)
        if year < 1900 or year > 2100:
            raise ValueError
    except ValueError:
        print("Enter a valid model year.")
        return

    lookup_recalls(make, model, year)


if __name__ == "__main__":
    main()
