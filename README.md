# Local RAG System with PostgreSQL & Ollama

This project implements a local **Retrieval-Augmented Generation (RAG)** system developed for the **Database Systems** course. It demonstrates the integration of relational databases with modern AI workflows, specifically focusing on vector similarity search and hybrid retrieval techniques.

## 🚀 Overview

The system allows users to ingest PDF documents into a PostgreSQL database and then query them using natural language. It leverages **pgvector** for vector storage and **Ollama** for local embeddings and LLM generation, ensuring that all data stays on the local machine.

## 🛠️ Technical Stack

- **Database**: [PostgreSQL](https://www.postgresql.org/) (v17+)
- **Vector Extension**: [pgvector](https://github.com/pgvector/pgvector) (HNSW Indexing)
- **Embeddings**: `nomic-embed-text` (via Ollama)
- **LLM**: `llama3.2` (via Ollama)
- **Language**: Python 3.x
- **Libraries**: `psycopg2`, `pypdf`, `pgvector-python`

## 🏗️ Key Database Features

1. **Vector Similarity Search**: Uses the `pgvector` extension to perform Approximate Nearest Neighbor (ANN) search using the **HNSW** (Hierarchical Navigable Small World) algorithm.
2. **Hybrid Search**: Combines semantic vector search with traditional PostgreSQL **Full-Text Search (FTS)** using a GIN index and `ts_rank_cd`.
3. **Custom Weighting**: Implements a **70:30 weighted re-ranking** logic, prioritizing semantic context (70%) while maintaining technical keyword precision (30%).
4. **Generated Columns**: Utilizes PostgreSQL's `GENERATED ALWAYS AS` feature to automatically maintain a `tsvector` for efficient keyword indexing.

## 📋 Installation

### 1. Prerequisites
- [PostgreSQL](https://www.postgresql.org/download/) installed and running.
- [Ollama](https://ollama.com/) installed.

### 2. Database Setup
Create the target database:
```bash
psql -U postgres -c "CREATE DATABASE databasesystems;"
```

### 3. Model Setup
Pull the required models via Ollama:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 4. Application Setup
```bash
# Clone the repository and enter the directory
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

### Ingesting Data
To process a PDF and store it in the database:
```bash
python rag.py --ingest path/to/your-document.pdf
```

### Querying
To start the interactive chat session:
```bash
python rag.py
```

## 📖 Architecture Detail
For a deep dive into the data flow, indexing strategies, and architectural decisions, please refer to the [GEMINI.md](./GEMINI.md) file.

---
**Course**: Database Systems  
**Project**: Local Retrieval-Augmented Generation System
