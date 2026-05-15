# Simple RAG System Architecture

This project implements a local Retrieval-Augmented Generation (RAG) system using PostgreSQL (pgvector) and Ollama.

## Architecture Overview

The system follows a standard RAG pipeline, optimized with **Hybrid Search** (Vector + Keyword) for higher accuracy.

### 1. Tools & Technologies
- **LLM/Embeddings**: [Ollama](https://ollama.com/) running locally.
  - **Generation**: `llama3.2`
  - **Embeddings**: `nomic-embed-text` (768 dimensions)
- **Vector Database**: [PostgreSQL](https://www.postgresql.org/) (v17+) with the [pgvector](https://github.com/pgvector/pgvector) extension.
- **PDF Processing**: `pypdf` for reliable text extraction.
- **Database Driver**: `psycopg2` with `pgvector-python`.

---

## Data Flow

### Phase A: Ingestion (`python rag.py --ingest [path_to_pdf]`)
1. **Extraction**: `pypdf` reads the PDF and extracts text along with page metadata.
2. **Chunking**: Text is split into segments (~1000 characters) with a 100-character overlap to preserve context across chunks.
3. **Embedding**: Chunks are converted into 768-dimensional vectors using `nomic-embed-text`.
4. **Storage**: Data is stored in the `document_chunks` table:
    - `content`: Raw text chunk.
    - `embedding`: Vector representation.
    - `metadata`: JSONB containing `page` and `source`.
    - `fulltext_tokens`: A `GENERATED` column for PostgreSQL Full-Text Search.
5. **Indexing**: 
    - **HNSW Index**: For fast vector similarity search.
    - **GIN Index**: For fast keyword/full-text search.

### Phase B: Retrieval & Generation (`python rag.py`)
1. **Hybrid Search**: When a user asks a question:
    - **Vector Search**: Finds the top 20 chunks by cosine similarity.
    - **Keyword Search**: Finds the top 20 chunks using BM25-like ranking (`ts_rank_cd`).
    - **Re-ranking**: Results are combined and ranked based on a composite score.
2. **Context Construction**: The top-K chunks are bundled into a prompt.
3. **Generation**: Ollama's `llama3.2` generates a response restricted to the provided context.
4. **Output**: The answer is printed with source citations (e.g., `vector-database.pdf (Page 5)`).

---

## Installation & Setup

1. **Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Database**:
   Ensure PostgreSQL is running and the `databasesystems` database exists:
   ```bash
   psql -c "CREATE DATABASE databasesystems;"
   ```

3. **Ollama**:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

---

## Usage

### Ingest Data
```bash
# Default file (vector-database.pdf)
python rag.py --ingest

# Specific file
python rag.py --ingest path/to/your.pdf
```

### Start Chat
```bash
python rag.py
```

---

## Restoration Notes (May 2026)
- **Schema Evolution**: Robustly handled the addition of `fulltext_tokens` to existing tables via `ALTER TABLE` logic in `init_db`.
- **Dependency Alignment**: Standardized on `pypdf` to match documentation mandates.
- **Performance**: Optimized retrieval with a Hybrid Search pattern to overcome limitations of pure vector search in technical documentation.
