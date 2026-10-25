# Vehicle Knowledge Assistant

**Note:** Start the FastAPI server locally before opening the application link. The local URL is accessible only on your computer.


An AI-powered vehicle recall assistant combining Retrieval-Augmented Generation (RAG), semantic search, a local language model, and live recall information from the NHTSA API.

## Features

- Ask natural-language questions about vehicle recalls.
- Retrieve relevant records using Sentence Transformers and FAISS.
- Generate answers using a local Ollama language model.
- Look up live vehicle recalls through the NHTSA API.
- Use a FastAPI web interface.
- Run automated tests and retrieval evaluations.

## Technology Stack

- Python
- FastAPI and Uvicorn
- Sentence Transformers
- FAISS
- Ollama
- NHTSA Recalls API
- pytest

## Project Structure

- `assistant.py` — command-line RAG assistant
- `nhtsa_agent.py` — vehicle recall lookup agent
- `search.py` — semantic search utilities
- `prepare_data.py` — data preparation
- `evaluate.py` — retrieval evaluation
- `main.py` — main project entry point
- `web_app.py` — FastAPI web application
- `test_web_app.py` — web application tests
- `data/` — local recall datasets

## Prerequisites

- Python 3.9 or a compatible version
- Ollama installed locally
- Internet access for live NHTSA lookups

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
