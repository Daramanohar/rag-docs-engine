# Form Processing RAG App

A Retrieval-Augmented Generation (RAG) pipeline built with Streamlit, PostgreSQL (with pgvector), and Llama 3 via Groq API. This app allows users to upload documents, extract and chunk text, embed with sentence-transformers, store in PostgreSQL, and retrieve relevant chunks for LLM-based Q&A.

---

## Features
- **Document Upload:** Supports PDF, DOCX, images, and text files.
- **Text Extraction:** OCR for images, parsing for PDFs/DOCX/text.
- **Chunking & Embedding:** Splits text and generates 384-dim embeddings using sentence-transformers.
- **Vector Database:** Stores embeddings in PostgreSQL with pgvector extension.
- **Semantic Search:** Retrieves relevant chunks using vector similarity.
- **LLM Q&A:** Uses Llama 3 via Groq API for question answering over retrieved context.
- **Named Entity Recognition:** Extracts and displays entities in both JSON and human-readable formats.
- **Data Management:** Sidebar button to clear all data (for research phase).

---

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **Database:** PostgreSQL (with pgvector, e.g., Supabase)
- **Embeddings:** sentence-transformers
- **LLM:** Llama 3 via Groq API
- **OCR:** Mistral API (for images)

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Daramanohar/rag-docs-engine/tree/main
   cd rag-docs-engine
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   - Add your API keys and database credentials to `.streamlit/secrets.toml` or the Streamlit Cloud secrets UI:
     ```toml
     MISTRAL_API_KEY = "your-mistral-key"
     GROQ_API_KEY = "your-groq-key"
     DB_HOST = "your-db-host"
     DB_NAME = "your-db-name"
     DB_USER = "your-db-user"
     DB_PASSWORD = "your-db-password"
     DB_PORT = "5432"
     ```

4. **Enable pgvector extension** in your PostgreSQL database (if not already enabled):
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

5. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## Usage
- Upload a document (PDF, DOCX, image, or text).
- The app extracts and chunks the text, generates embeddings, and stores them in the database.
- Ask questions about the document in the chatbot tab.
- View named entities in both JSON and human-readable formats.
- Use the sidebar to clear all data (for research/testing phase).

---

## Deployment
- Deploy on [Streamlit Cloud](https://streamlit.io/cloud) or your own server.
- Set secrets in the Streamlit Cloud UI for API keys and DB credentials.
- Make sure your database is accessible from the deployment environment.

---

## Notes
- **Research Phase:** The "Clear All Data" button wipes the entire database for all users. This is simple for prototyping, but not suitable for production.
- **Production Recommendation:** For multi-user support, store a user/session ID with each chunk and filter queries/clearing by this ID.
- **OCR Model:** Requires Mistral API for image OCR.
- **LLM Model:** Uses Llama 3 via Groq API for Q&A.

---

## Future Improvements
- User/session-specific chunk storage and clearing
- User authentication
- More robust error handling and logging
- Support for additional file types and languages

---
