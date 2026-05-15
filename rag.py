import os
import sys
import json
import psycopg2
import numpy as np
import ollama
from pypdf import PdfReader
from pgvector.psycopg2 import register_vector

# Configuration
DB_URL = "postgresql://localhost:5432/databasesystems"
PDF_PATH = "vector-database.pdf"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"

def init_db():
    """Initialize the database with pgvector and full-text search support."""
    print("Initializing database...")
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table with all columns from the start
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id bigserial PRIMARY KEY,
                    content text NOT NULL,
                    embedding vector(768),
                    metadata jsonb,
                    fulltext_tokens tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                );
            """)
            
            # Robustly add column if it was missing from an old table version
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name='document_chunks' AND column_name='fulltext_tokens') THEN
                        ALTER TABLE document_chunks ADD COLUMN fulltext_tokens tsvector 
                        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
                    END IF;
                EXCEPTION
                    WHEN others THEN
                        NULL;
                END $$;
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx ON document_chunks USING hnsw (embedding vector_cosine_ops);")
            cur.execute("CREATE INDEX IF NOT EXISTS document_chunks_fulltext_idx ON document_chunks USING GIN (fulltext_tokens);")
        conn.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def extract_text_and_metadata(pdf_path):
    """Extract text from a PDF file using pypdf."""
    filename = os.path.basename(pdf_path)
    print(f"Extracting text and metadata from {filename}...")
    pages_data = []
    
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and len(page_text.strip()) > 50:
                pages_data.append({
                    "text": page_text,
                    "page_label": i + 1,
                    "filename": filename
                })
    except Exception as e:
        print(f"Error reading PDF: {e}")
        
    return pages_data

def chunk_data(pages_data, chunk_size=1000, overlap=100):
    """Split text into chunks with overlap, filtering out low-quality chunks."""
    chunks = []
    for page in pages_data:
        text = page["text"]
        # Basic cleanup: remove multiple newlines and spaces
        text = " ".join(text.split())
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Only keep chunks that have a reasonable amount of text
            if len(chunk_text.strip()) > 100:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "page": page["page_label"],
                        "source": page["filename"]
                    }
                })
            start += chunk_size - overlap
    return chunks

def get_embeddings(chunks):
    """Get embeddings for a list of chunks using Ollama."""
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = []
    for chunk in chunks:
        response = ollama.embed(model=EMBED_MODEL, input=chunk["content"])
        embeddings.append(response['embeddings'][0])
    return embeddings

def store_chunks(chunks, embeddings, clear_existing=False):
    """Store chunks and their embeddings in PostgreSQL."""
    print("Storing chunks in database...")
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        register_vector(conn)
        
        with conn.cursor() as cur:
            if clear_existing:
                print("Clearing existing data...")
                cur.execute("TRUNCATE TABLE document_chunks;")
            
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    "INSERT INTO document_chunks (content, embedding, metadata) VALUES (%s, %s, %s)",
                    (chunk["content"], np.array(embedding), json.dumps(chunk["metadata"]))
                )
        conn.close()
        print("Data storage complete.")
    except Exception as e:
        print(f"Error storing chunks: {e}")

def query_rag(query, top_k=5):
    """Retrieve relevant chunks using Hybrid Search (Vector + Full-Text)."""
    # 1. Get embedding for the query
    query_embed = ollama.embed(model=EMBED_MODEL, input=query)['embeddings'][0]
    
    # 2. Search in PostgreSQL
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    
    with conn.cursor() as cur:
        # Hybrid Search: Combine Vector Similarity and Full-Text Search
        hybrid_query = """
        WITH vector_search AS (
            SELECT id, content, metadata, 1 - (embedding <=> %s) as similarity
            FROM document_chunks
            ORDER BY embedding <=> %s
            LIMIT 20
        ),
        keyword_search AS (
            SELECT id, content, metadata, ts_rank_cd(fulltext_tokens, plainto_tsquery('english', %s)) as rank
            FROM document_chunks
            WHERE fulltext_tokens @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT 20
        )
        SELECT 
            COALESCE(v.content, k.content) as content,
            COALESCE(v.metadata, k.metadata) as metadata,
            COALESCE(v.similarity, 0) + COALESCE(k.rank, 0) as score
        FROM vector_search v
        FULL OUTER JOIN keyword_search k ON v.id = k.id
        ORDER BY score DESC
        LIMIT %s;
        """
        cur.execute(hybrid_query, (np.array(query_embed), np.array(query_embed), query, query, top_k))
        results = cur.fetchall()
    conn.close()
    
    context_parts = []
    sources = []
    for content, metadata, score in results:
        context_parts.append(content)
        if metadata:
            source_info = f"{metadata.get('source', 'Unknown')} (Page {metadata.get('page', '?')})"
            sources.append(source_info)
    
    context = "\n\n".join(context_parts)
    source_str = " | ".join(set(sources))
    
    # 3. Generate response with Ollama
    prompt = f"""
    You are a helpful assistant. Use the following context to answer the user's question.
    If the answer is not in the context, say that you don't know based on the provided document.
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """
    
    response = ollama.chat(model=CHAT_MODEL, messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    return response['message']['content'], source_str

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ingest":
        ingest_path = sys.argv[2] if len(sys.argv) > 2 else PDF_PATH
        
        if not os.path.exists(ingest_path):
            print(f"Error: File {ingest_path} not found.")
            sys.exit(1)
            
        init_db()
        pages_data = extract_text_and_metadata(ingest_path)
        chunks = chunk_data(pages_data)
        embeddings = get_embeddings(chunks)
        
        clear = input("Clear existing database records? (y/N): ").lower() == 'y'
        store_chunks(chunks, embeddings, clear_existing=clear)
        print(f"Ingestion of {os.path.basename(ingest_path)} finished.")
    else:
        print("Welcome to local RAG! (Type 'exit' to quit)")
        while True:
            user_input = input("\nYour Question: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            try:
                answer, sources = query_rag(user_input)
                print(f"\nRAG Answer: {answer}")
                print(f"\nSources: {sources}")
            except Exception as e:
                print(f"Error querying: {e}")
                print("Tip: If the database is empty, run: python rag.py --ingest")
