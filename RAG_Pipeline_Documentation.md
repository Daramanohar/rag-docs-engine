# RAG Pipeline Documentation

## 1. High-Level Workflow

1. **User uploads a document** (image, PDF, DOCX, or text) via a Streamlit web interface.
2. **Text is extracted** from the document using OCR (for images), PyMuPDF (for PDFs), python-docx (for DOCX), or direct reading (for text files).
3. **Text is chunked** into small, focused pieces (default: 5 sentences per chunk).
4. **Each chunk is embedded** into a 384-dimensional vector using the `sentence-transformers/all-MiniLM-L6-v2` model.
5. **Chunks and embeddings are stored** in a PostgreSQL database with the `pgvector` extension for vector search.
6. **User asks a question** in the chatbot interface.
7. **The app retrieves the most relevant chunks** from the database using vector similarity search.
8. **The retrieved context and user question are sent to Llama 3-70B** (via Groq API) for answer generation.
9. **The answer is displayed** to the user in the Streamlit app.

---

## 2. Technology Stack

| Component         | Technology/Library                                    | Purpose                                      |
|-------------------|-------------------------------------------------------|----------------------------------------------|
| UI                | Streamlit                                             | Web interface for upload, chat, analytics    |
| OCR               | Mistral API, PIL                                      | Extract text from images                     |
| PDF Extraction    | PyMuPDF (`fitz`)                                      | Extract text from PDFs                       |
| DOCX Extraction   | python-docx                                           | Extract text from DOCX files                 |
| Text Embedding    | sentence-transformers (`all-MiniLM-L6-v2`)            | Generate 384-dim vector embeddings           |
| Database          | PostgreSQL + pgvector                                 | Store chunks and embeddings, vector search   |
| LLM               | Llama 3-70B via Groq API                              | Summarization, Q&A, key-value extraction     |
| NLP               | spaCy                                                 | Entity extraction, text processing           |
| Backend           | Python, psycopg2                                      | Glue logic, DB access                        |
| Logging           | Python logging                                        | Error and info logging                       |

---

## 3. File-by-File Breakdown

### A. `streamlit_app.py`
- **Role:** Main entry point and UI for the app.
- **Key Functions:**
  - Handles file uploads (image, PDF, DOCX, text).
  - Extracts text using the appropriate method.
  - Calls the processing pipeline (chunking, embedding, storage).
  - Provides a chatbot interface for user Q&A.
  - Manages session state and data clearing (including DB).
  - Displays results, summaries, and analytics.
- **Technologies Used:** Streamlit, PIL, fitz, docx, custom modules.

### B. `modules/rag_utils.py`
- **Role:** Core RAG pipeline utilities.
- **Key Functions:**
  - `get_db_conn()`: Connects to PostgreSQL.
  - `create_documents_table()`: Ensures the documents table and pgvector extension exist.
  - `chunk_text()`: Splits text into small chunks (default: 5 sentences).
  - `get_embedding()`: Generates 384-dim embeddings using sentence-transformers.
  - `insert_chunk()`: Stores a chunk and its embedding in the DB.
  - `fetch_similar_chunks()`: Retrieves top-k similar chunks using vector similarity.
  - `store_processed_doc_in_rag()`: Orchestrates chunking, embedding, and storage for a document.
  - `clear_documents_table()`: Deletes all rows from the documents table.
- **Technologies Used:** psycopg2, sentence-transformers, spaCy, regex, logging.

### C. `modules/chatbot.py`
- **Role:** Handles chatbot logic and LLM prompt construction.
- **Key Functions:**
  - `build_rag_prompt()`: Builds a prompt for Llama using retrieved context and user question, with debug logging.
  - `get_llama_rag_response()`: Calls Groq API with the prompt and returns the Llama-generated answer.
- **Technologies Used:** Groq API, logging, custom RAG utilities.

### D. `modules/ocr_processor.py`
- **Role:** Handles OCR for image files.
- **Key Functions:**
  - Encodes images to base64.
  - Calls Mistral OCR API.
  - Extracts and post-processes text from OCR results.
  - Identifies form type based on content/filename.
- **Technologies Used:** Mistral API, PIL, base64, custom form utilities.

### E. `modules/data_analyzer.py`
- **Role:** Analyzes extracted text for key-value pairs, summary, and completeness.
- **Key Functions:**
  - `analyze_document()`: Extracts key-values, generates summary, checks completeness.
  - Uses regex, NLP, and LLM for analysis.
- **Technologies Used:** spaCy, regex, LLM (Groq), logging.

### F. `modules/form_utils.py`
- **Role:** Utility functions for form processing and validation.
- **Key Functions:**
  - Timestamp generation.
  - Document data validation.
  - Search and preview formatting for processed documents.
- **Technologies Used:** pandas, datetime, json, csv.

### G. `requirements.txt`
- **Role:** Lists all Python dependencies for the project.
- **Key Libraries:** streamlit, groq, mistralai, pandas, pillow, python-dotenv, requests, pydantic, psycopg2-binary, spacy, PyMuPDF, sentence-transformers, python-docx.

---

## 4. Data Flow Diagram

```mermaid
flowchart TD
    A[User uploads document] --> B[Text extraction (OCR/PDF/DOCX/TXT)]
    B --> C[Chunking (5 sentences per chunk)]
    C --> D[Embedding (384-dim vector)]
    D --> E[Store in PostgreSQL + pgvector]
    F[User asks question] --> G[Embed question]
    G --> H[Vector search: fetch similar chunks]
    H --> I[Build prompt with context + question]
    I --> J[Llama 3-70B via Groq API]
    J --> K[Display answer in Streamlit UI]
```

---

## 5. Key Features and Best Practices

- Efficient vector search using pgvector for fast, semantic retrieval.
- Small, focused chunks for high-relevance context.
- LLM-powered summarization and Q&A with context injection (RAG).
- Robust file support: images (OCR), PDFs, DOCX, and text.
- Easy data clearing (session and database).
- Debug logging for context and prompt inspection.
- Modular codebase for easy maintenance and extension.

---

## 6. How Each File Contributes

| File/Module           | Main Responsibility                                      |
|-----------------------|---------------------------------------------------------|
| streamlit_app.py      | UI, file handling, user interaction, main workflow      |
| modules/rag_utils.py  | Chunking, embedding, DB storage/retrieval, clearing     |
| modules/chatbot.py    | Prompt building, LLM API calls, debug logging           |
| modules/ocr_processor.py | OCR for images, form type detection                  |
| modules/data_analyzer.py | Key-value extraction, summarization, completeness    |
| modules/form_utils.py | Validation, search, formatting, timestamps              |
| requirements.txt      | Dependency management                                   |

---

## 7. Technologies Used

- **Frontend/UI:** Streamlit
- **Backend:** Python
- **Database:** PostgreSQL with pgvector extension
- **LLM:** Llama 3-70B via Groq API
- **OCR:** Mistral API, PIL
- **PDF:** PyMuPDF
- **DOCX:** python-docx
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **NLP:** spaCy
- **DB Access:** psycopg2
- **Logging:** Python logging

---

## 8. Extensibility

- Add more file types by extending the upload and extraction logic.
- Tune chunking (by tokens, paragraphs, etc.) for different use cases.
- Add analytics or dashboards using Streamlit.
- Swap LLMs or embedding models as needed.

---

# Conclusion

Your pipeline is a modern, modular RAG system that leverages the best of open-source and cloud AI:
- User-friendly UI
- Robust document processing
- Semantic search and LLM-powered answers
- Easy to maintain and extend 