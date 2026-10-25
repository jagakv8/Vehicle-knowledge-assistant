import json
import os
import re

import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "recall_documents.json")

NHTSA_URL = "https://api.nhtsa.gov/recalls/recallsByVehicle"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4
MIN_SIMILARITY = 0.30

app = FastAPI(title="Vehicle Knowledge Assistant")

embedder = None
documents = []
search_index = None


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vehicle Knowledge Assistant</title>
<style>
body {
  font-family: -apple-system, BlinkMacSystemFont, Arial, sans-serif;
  max-width: 900px; margin: 35px auto; padding: 20px;
  background: #f4f7fb; color: #172033;
}
h1 { margin-bottom: 8px; }
.subtitle { color: #586174; }
form, .card {
  background: white; padding: 22px; border-radius: 14px;
  box-shadow: 0 4px 18px #17203312; margin: 18px 0;
}
label { display: block; margin-top: 12px; }
input, textarea {
  box-sizing: border-box; width: 100%; padding: 12px;
  margin-top: 6px; border: 1px solid #ccd3df;
  border-radius: 8px; font-size: 15px;
}
textarea { min-height: 100px; resize: vertical; }
button {
  margin-top: 14px; background: #2158d5; color: white;
  padding: 12px 18px; border: 0; border-radius: 8px;
  cursor: pointer; font-size: 15px;
}
button:disabled { opacity: .6; }
.campaign { color: #2158d5; font-weight: bold; }
.muted { color: #586174; }
#error { color: #b42318; white-space: pre-wrap; }
#answer { white-space: pre-wrap; line-height: 1.6; }
</style>
</head>
<body>
<h1>Vehicle Knowledge Assistant</h1>
<p class="subtitle">
Ask a general recall question or look up recalls for a specific vehicle.
</p>

<form id="questionForm">
  <label for="question">Your question</label>
  <textarea id="question" required
    placeholder="What recalls affect a 2020 Toyota Camry?&#10;What happens if a fuel pump fails?"></textarea>
  <button id="submit" type="submit">Ask assistant</button>
</form>

<p id="status" class="muted"></p>
<p id="error"></p>
<div id="answerCard" class="card" hidden>
  <h2>Answer</h2>
  <div id="answer"></div>
</div>
<div id="evidence"></div>
<p class="muted">
Vehicle recall results do not confirm VIN-level eligibility or repair status.
Verify with NHTSA or an authorized dealer.
</p>

<script>
const form = document.getElementById("questionForm");
const button = document.getElementById("submit");
const statusText = document.getElementById("status");
const errorText = document.getElementById("error");
const answerCard = document.getElementById("answerCard");
const answerBox = document.getElementById("answer");
const evidenceBox = document.getElementById("evidence");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  statusText.textContent = "Searching evidence and preparing an answer...";
  errorText.textContent = "";
  answerCard.hidden = true;
  evidenceBox.replaceChildren();

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        question: document.getElementById("question").value.trim()
      })
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Request failed.");

    statusText.textContent = "Source: " + data.mode;
    answerBox.textContent = data.answer;
    answerCard.hidden = false;

    for (const item of (data.evidence || [])) {
      const card = document.createElement("section");
      card.className = "card";

      const heading = document.createElement("h3");
      heading.textContent = item.component || "Recall evidence";
      card.appendChild(heading);

      const campaign = document.createElement("p");
      campaign.className = "campaign";
      campaign.textContent = "Campaign: " + (item.campaign || "Unavailable");
      card.appendChild(campaign);

      for (const [label, value] of [
        ["Issue", item.summary],
        ["Consequence", item.consequence],
        ["Remedy", item.remedy],
        ["Relevance score", item.score == null ? "" : item.score.toFixed(3)]
      ]) {
        if (!value) continue;
        const p = document.createElement("p");
        const strong = document.createElement("strong");
        strong.textContent = label + ": ";
        p.appendChild(strong);
        p.appendChild(document.createTextNode(String(value)));
        card.appendChild(p);
      }
      evidenceBox.appendChild(card);
    }
  } catch (error) {
    errorText.textContent = error.message;
    statusText.textContent = "";
  } finally {
    button.disabled = false;
  }
});
</script>
</body>
</html>
"""


class AskRequest(BaseModel):
    question: str


def get_campaign(doc):
    value = str(doc.get("id", ""))
    match = re.search(r"\b\d{2}V\d{6}\b", value, re.I)
    if not match:
        match = re.search(r"\b\d{2}V\d{6}\b", str(doc.get("text", "")), re.I)
    return match.group(0).upper() if match else None


def deduplicate(items):
    unique = []
    seen = set()
    for item in items:
        campaign = get_campaign(item)
        key = campaign or str(item.get("id", item.get("text", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def parse_vehicle_query(question):
    pattern = re.compile(
        r"\b(19\d{2}|20\d{2})\s+"
        r"([A-Za-z][A-Za-z0-9-]*)\s+"
        r"([A-Za-z0-9][A-Za-z0-9-]*"
        r"(?:\s+[A-Za-z0-9][A-Za-z0-9-]*){0,2})",
        re.I,
    )
    match = pattern.search(question)
    if not match:
        return None

    year, make, model = match.groups()
    model = re.split(
        r"\b(recalls?|problems?|issues?|defects?|vehicles?|cars?)\b",
        model,
        flags=re.I,
    )[0].strip()

    if not model:
        return None
    return int(year), make, model


def fetch_live_recalls(year, make, model):
    response = requests.get(
        NHTSA_URL,
        params={"modelYear": year, "make": make, "model": model},
        timeout=30,
    )
    response.raise_for_status()
    records = response.json().get("results", [])
    unique = {}

    for record in records:
        campaign = str(
            record.get("NHTSACampaignNumber")
            or record.get("CampaignNumber")
            or "Unknown"
        ).strip()

        if campaign in unique:
            continue

        unique[campaign] = {
            "campaign": campaign,
            "component": record.get("Component", ""),
            "summary": record.get("Summary", ""),
            "consequence": (
                record.get("Conequence") or record.get("Consequence") or ""
            ),
            "remedy": record.get("Remedy", ""),
        }

    return list(unique.values())


def search_local(question):
    if not documents or search_index is None or embedder is None:
        return []

    vector = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vector = np.asarray(vector, dtype=np.float32)
    count = min(TOP_K, len(documents))
    scores, indices = search_index.search(vector, count)

    results = []
    seen = set()

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or float(score) < MIN_SIMILARITY:
            continue

        doc = documents[int(idx)]
        campaign = get_campaign(doc)
        key = campaign or str(doc.get("id", ""))

        if key in seen:
            continue
        seen.add(key)

        text = str(doc.get("text", ""))
        results.append(
            {
                "campaign": campaign or "Unavailable",
                "component": "",
                "summary": text[:1800],
                "consequence": "",
                "remedy": "",
                "score": float(score),
                "source": "Local recall dataset",
            }
        )

    return results


def generate_local_answer(question, evidence):
    context = "\n\n".join(
        f"Campaign: {item['campaign']}\nDetails: {item['summary']}" for item in evidence
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful vehicle recall assistant. Answer "
                        "concisely using only the supplied evidence. Do not "
                        "invent vehicle eligibility, campaign IDs, or remedies. "
                        "Say when the evidence is insufficient. For general "
                        "mechanical questions, distinguish general information "
                        "from recall-specific findings. Cite only supplied "
                        "campaign IDs. Do not list every affected model unless asked."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nEvidence:\n{context}",
                },
            ],
            "options": {"temperature": 0, "num_ctx": 8192},
        },
        timeout=180,
    )
    response.raise_for_status()
    answer = response.json().get("message", {}).get("content", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    allowed = {item["campaign"] for item in evidence}
    for cited in set(re.findall(r"\b\d{2}V\d{6}\b", answer, re.I)):
        if cited.upper() not in allowed:
            answer = re.sub(
                re.escape(cited),
                "[unverified campaign reference removed]",
                answer,
                flags=re.I,
            )

    return answer


@app.on_event("startup")
def load_rag():
    global embedder, documents, search_index

    if not os.path.exists(DATA_FILE):
        print("Local recall dataset not found; live vehicle lookup remains available.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        loaded = json.load(file)

    documents = deduplicate(
        [doc for doc in loaded if isinstance(doc, dict) and doc.get("text")]
    )

    if not documents:
        print("Local recall dataset is empty.")
        return

    print("Loading local embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    vectors = embedder.encode(
        [str(doc["text"]) for doc in documents],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    search_index = faiss.IndexFlatIP(vectors.shape[1])
    search_index.add(vectors)

    print("Local RAG ready:", len(documents), "unique records.")


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.get("/health")
def health():
    return {
        "status": "ok",
        "local_rag_ready": search_index is not None,
        "local_records": len(documents),
    }


@app.post("/api/ask")
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    vehicle = parse_vehicle_query(question)

    if vehicle:
        year, make, model = vehicle
        try:
            evidence = fetch_live_recalls(year, make, model)
        except requests.RequestException as error:
            raise HTTPException(
                status_code=502,
                detail="NHTSA lookup failed. Please try again shortly.",
            ) from error

        if not evidence:
            return {
                "mode": "Live NHTSA lookup",
                "answer": (
                    f"NHTSA returned no records for {year} {make} {model}. "
                    "This does not prove that no recall exists. Verify by VIN."
                ),
                "evidence": [],
            }

        lines = [
            f"NHTSA returned {len(evidence)} unique campaign record(s) "
            f"for the {year} {make} {model} lookup."
        ]

        for item in evidence:
            lines.append(
                f"\nCampaign {item['campaign']}\n"
                f"Component: {item['component'] or 'Not provided'}\n"
                f"Issue: {item['summary'] or 'Not provided'}\n"
                f"Consequence: {item['consequence'] or 'Not provided'}\n"
                f"Remedy: {item['remedy'] or 'Not provided'}"
            )

        lines.append(
            "\nVerify eligibility and repair status by VIN with NHTSA "
            "or an authorized dealer."
        )

        return {
            "mode": "Live NHTSA lookup",
            "answer": "\n".join(lines),
            "evidence": evidence,
        }

    evidence = search_local(question)
    if not evidence:
        return {
            "mode": "Local RAG",
            "answer": (
                "I couldn't find sufficiently relevant records in the local "
                "recall dataset. This does not mean no recall exists."
            ),
            "evidence": [],
        }

    try:
        answer = generate_local_answer(question, evidence)
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail="Could not reach Ollama. Make sure Ollama is running.",
        ) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {
        "mode": "Local RAG + Ollama",
        "answer": answer,
        "evidence": evidence,
    }
