from sentence_transformers import SentenceTransformer
import spacy
import re
import psycopg2
import logging

# Initialize models
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Database configuration
DB_CONFIG = {
    "dbname": "formprocessing",
    "user": "postgres",
    "password": "5657",  # Make sure this matches your actual password
    "host": "localhost",
    "port": 5432
}

def get_db_conn():
    """Get database connection"""
    print("DB_CONFIG being used:", DB_CONFIG)
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected to:", conn.get_dsn_parameters())
        return conn
    except Exception as e:
        print("Database connection failed:", e)
        logging.error(f"Database connection failed: {e}")
        return None

def create_documents_table():
    """Create the documents table if it doesn't exist"""
    conn = get_db_conn()
    if conn is None:
        return False
    try:
        with conn:
            with conn.cursor() as cur:
                # Create extension if not exists
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create table if not exists
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS public.documents (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector(384),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create index for faster similarity search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                    ON public.documents USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = 100);
                """)
                
                print("✅ Documents table created/verified successfully")
                return True
    except Exception as e:
        print(f"❌ Failed to create documents table: {e}")
        logging.error(f"Failed to create documents table: {e}")
        return False
    finally:
        conn.close()

def chunk_text(text, chunk_size=5):
    """Split text into chunks based on sentences"""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def get_embedding(text):
    """Generate embedding for text"""
    try:
        embedding = embedder.encode(text)
        return embedding.tolist()
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        print(f"❌ Embedding generation failed: {e}")
        return None

def postprocess(text):
    """Postprocess text and extract entities"""
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return text, entities

def insert_chunk(text, embedding):
    """Insert text chunk and its embedding into database"""
    conn = get_db_conn()
    if conn is None:
        print("❌ DB connection failed")
        return False
    
    try:
        with conn:
            with conn.cursor() as cur:
                # Debug: Print embedding info
                print(f"Embedding type: {type(embedding)}")
                print(f"Embedding length: {len(embedding) if hasattr(embedding, '__len__') else 'N/A'}")
                
                # Ensure embedding is a list
                if not isinstance(embedding, list):
                    embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                
                # Insert the chunk
                cur.execute(
                    "INSERT INTO public.documents (content, embedding) VALUES (%s, %s)",
                    (text, embedding)
                )
                
                # Get the inserted ID for confirmation
                cur.execute("SELECT lastval();")
                inserted_id = cur.fetchone()[0]
                print(f"✅ Inserted chunk successfully with ID: {inserted_id}")
                return True
                
    except Exception as e:
        print(f"❌ Failed to insert chunk: {e}")
        print(f"Text length: {len(text)}")
        print(f"Embedding sample: {embedding[:5] if embedding else 'None'}")
        logging.error(f"Failed to insert chunk: {e}")
        return False
    finally:
        conn.close()

def fetch_similar_chunks(query_embedding, top_k=3):
    """Fetch similar chunks from database using vector similarity"""
    conn = get_db_conn()
    if conn is None:
        return []
    try:
        with conn:
            with conn.cursor() as cur:
                # Ensure query_embedding is a list
                if not isinstance(query_embedding, list):
                    query_embedding = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
                # Convert embedding to pgvector string format
                embedding_str = str(query_embedding)
                cur.execute(
                    """
                    SELECT content, embedding <-> %s::vector AS distance
                    FROM public.documents
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                    """,
                    (embedding_str, embedding_str, top_k)
                )
                results = cur.fetchall()
                print(f"✅ Found {len(results)} similar chunks")
                return [row[0] for row in results]
    except Exception as e:
        logging.error(f"Failed to fetch similar chunks: {e}")
        print(f"❌ Failed to fetch similar chunks: {e}")
        return []
    finally:
        conn.close()

def store_processed_doc_in_rag(doc):
    """Store processed document in RAG vector database"""
    text_to_chunk = doc['ocr_result']['text']
    chunks = chunk_text(text_to_chunk)
    
    successful_chunks = 0
    failed_chunks = 0
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
        
        try:
            embedding = get_embedding(chunk)
            if embedding is not None:
                if insert_chunk(chunk, embedding):
                    successful_chunks += 1
                else:
                    failed_chunks += 1
            else:
                print(f"❌ Failed to generate embedding for chunk {i+1}")
                failed_chunks += 1
        except Exception as e:
            print(f"❌ Error processing chunk {i+1}: {e}")
            failed_chunks += 1
    
    print(f"📊 Storage complete: {successful_chunks} successful, {failed_chunks} failed")
    return successful_chunks, failed_chunks

def clear_documents_table():
    conn = get_db_conn()
    if conn is None:
        return False
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM public.documents;")
                print("✅ All documents deleted from database.")
                return True
    except Exception as e:
        print(f"❌ Failed to clear documents table: {e}")
        return False
    finally:
        conn.close()
