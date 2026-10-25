# 🚗 Vehicle Knowledge Assistant

## Project Overview

The Vehicle Knowledge Assistant is an AI-powered web application designed to help users understand vehicle recalls and access relevant automotive safety information. It combines Retrieval-Augmented Generation (RAG), semantic search, a large language model, and live data from the National Highway Traffic Safety Administration (NHTSA).

The application retrieves relevant information from a local recall knowledge base and uses an AI model to generate concise, evidence-based answers. It can also look up live recall information for specific vehicle makes, models, and model years.

## 🎯 Project Objectives

- Make vehicle recall information easier to understand.
- Retrieve relevant information from a structured knowledge base.
- Generate answers grounded in available evidence.
- Integrate live vehicle recall data from the NHTSA API.
- Reduce unsupported claims by validating cited recall campaign numbers.
- Demonstrate practical applications of RAG, semantic search, and AI integration.

## ✨ Key Features

### 🔎 Semantic Search and RAG
- Converts recall documents into vector embeddings.
- Uses FAISS for similarity-based retrieval.
- Retrieves relevant documents for a user's question.
- Applies a similarity threshold to reduce irrelevant results.

### 🤖 AI-Powered Answers
- Uses the Qwen 2.5 3B model through Ollama.
- Generates concise responses using retrieved evidence.
- Handles insufficient evidence rather than automatically inventing an answer.
- Validates campaign references against the supplied evidence.

### 🌐 Live NHTSA Recall Lookup
- Retrieves recall records from the NHTSA API.
- Supports queries containing a vehicle's model year, make, and model.
- Displays campaign numbers, affected components, safety issues, consequences, and remedies when available.
- Reminds users to verify recall eligibility by VIN with NHTSA or an authorized dealer.

### 💻 Web Application
- Built with FastAPI.
- Provides a browser-based question-and-answer interface.
- Displays answers and supporting recall evidence.
- Includes a health endpoint for checking application status.

### 🧪 Evaluation and Testing
- Evaluates retrieval quality using predefined test queries.
- Includes automated tests with Pytest.
- Supports verification of application functionality and API behavior.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and data processing |
| FastAPI | Web API and backend |
| Sentence Transformers | Text embeddings |
| `all-MiniLM-L6-v2` | Embedding model |
| FAISS | Vector similarity search |
| Ollama | Local language-model inference |
| Qwen 2.5 3B | Answer generation |
| NHTSA API | Live vehicle recall information |
| NumPy | Numerical and vector operations |
| Pytest | Automated testing |

## ⚙️ How It Works

1. **Data preparation:** Recall documents are prepared and stored in JSON format.
2. **Embedding generation:** The embedding model converts document text into numerical vectors.
3. **Vector indexing:** FAISS indexes the vectors for efficient similarity search.
4. **Question processing:** The application receives a user's question and determines whether it contains a vehicle-specific recall lookup.
5. **Information retrieval:** General recall questions are matched against the local knowledge base.
6. **Answer generation:** Ollama generates a response using the retrieved evidence.
7. **Live lookup:** Vehicle-specific queries use the NHTSA API to retrieve current recall records.
8. **Response presentation:** The web interface displays the answer and relevant evidence.

## 🏗️ Architecture

**User → FastAPI Web Interface → Query Processing → RAG Retrieval / NHTSA API → Ollama (for RAG answers) → Response and Evidence**

The local RAG pipeline and live recall lookup are separate paths. Vehicle-specific queries can return NHTSA results directly, while general questions use local retrieval and the language model.

## 📂 Project Structure

```text
vehicle-knowledge-assistant/
├── assistant.py
├── evaluate.py
├── nhtsa_agent.py
├── search.py
├── main.py
├── prepare_data.py
├── web_app.py
├── test_web_app.py
├── requirements.txt
├── README.md
└── data/
    ├── recall_documents.json
    └── recalls.json
```

## 🚀 Running Locally

### Prerequisites
- Python
- Ollama
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/jagakv8/Vehicle-knowledge-assistant.git
cd Vehicle-knowledge-assistant
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure Ollama is installed and running, then download the required model:

```bash
ollama pull qwen2.5:3b
```

Start the application:

```bash
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Open the application in your browser:

`http://127.0.0.1:8000/`

The local address works only on the computer running the application. A public live demo requires deployment.

## 🧪 Testing

Run the automated tests:

```bash
python -m pytest -v
```

Run the retrieval evaluation:

```bash
python evaluate.py
```

## 🔐 Reliability and Safety

- Uses retrieved evidence to ground AI-generated answers.
- Filters unsupported recall campaign references.
- Returns an insufficient-evidence response when relevant records cannot be retrieved.
- Distinguishes recall findings from general mechanical information.
- Advises users to verify vehicle-specific recall eligibility by VIN.

This application is an informational tool and does not replace official recall notices, vehicle inspections, or advice from qualified automotive professionals.

## 🔮 Future Improvements

- Public cloud deployment and a live demo.
- Expanded recall datasets and improved retrieval evaluation.
- More robust vehicle entity extraction.
- VIN-based recall eligibility integration where supported by official services.
- Improved source citations and traceability.
- Enhanced monitoring, error handling, and user experience.

## 📌 Project Status

The project includes a working local web application, a RAG retrieval pipeline, integration with Ollama, live NHTSA recall lookup, and automated tests.

The application currently runs locally. Public deployment is a future improvement.

## 👨‍💻 Author

**Jagadeesh K V**

GitHub: https://github.com/jagakv8

---

*Built to explore practical applications of Retrieval-Augmented Generation, vector search, local language models, and real-world automotive safety data.*
