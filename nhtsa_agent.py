import requests

NHTSA_URL = "https://api.nhtsa.gov/recalls/recallsByVehicle"


def main():
    print("=" * 45)
    print("VEHICLE KNOWLEDGE ASSISTANT")
    print("Type 'exit' to quit.")
    print("=" * 45)

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        import re

        match = re.search(
            r"\b(19\d{2}|20\d{2})\s+([A-Za-z-]+)\s+([A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+){0,2})",
            question,
            re.I,
        )

        if not match:
            print("Please include a year, make, and model, e.g. 2020 Toyota Camry.")
            continue

        year, make, model = match.groups()
        model = re.split(r"\b(recalls?|issues?|problems?)\b", model, flags=re.I)[
            0
        ].strip()

        try:
            response = requests.get(
                NHTSA_URL,
                params={"modelYear": year, "make": make, "model": model},
                timeout=30,
            )
            response.raise_for_status()
            records = response.json().get("results", [])
            seen = set()
            unique = []

            for record in records:
                campaign = record.get("NHTSACampaignNumber", "Unknown")
                if campaign not in seen:
                    seen.add(campaign)
                    unique.append(record)

            print(f"\nNHTSA returned {len(unique)} unique recall record(s).")
            for i, record in enumerate(unique, 1):
                print(
                    f"\n{i}. Campaign: {record.get('NHTSACampaignNumber', 'Unknown')}"
                )
                print("Component:", record.get("Component", "Not provided"))
                print("Issue:", record.get("Summary", "Not provided"))
                print(
                    "Consequence:",
                    record.get("Conequence")
                    or record.get("Consequence", "Not provided"),
                )
                print("Remedy:", record.get("Remedy", "Not provided"))

            print(
                "\nVerify recall eligibility by VIN with NHTSA or an authorized dealer."
            )

        except requests.RequestException as error:
            print("NHTSA request failed:", error)


if __name__ == "__main__":
    main()
