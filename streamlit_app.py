import streamlit as st
import os
from PIL import Image
import json
import sys
import psycopg2
import spacy
import re
import logging
import fitz  # PyMuPDF for PDF text extraction
import docx
import io
from modules.chatbot import store_processed_doc_in_rag, get_llama_rag_response
from modules.rag_utils import chunk_text, get_embedding, postprocess, fetch_similar_chunks
from modules.form_utils import FormUtils

# Add this near the top of your streamlit_app.py, after imports
from modules.rag_utils import create_documents_table
from modules.rag_utils import clear_documents_table

# Initialize database table
@st.cache_resource
def init_database():
    """Initialize the database table"""
    success = create_documents_table()
    if success:
        st.success("Database initialized successfully!")
        return True
    else:
        st.error("Failed to initialize database!")
        return False

# Call this after your imports and before the main app logic
if not init_database():
    st.stop()

DB_CONFIG = {
    "dbname": "formprocessing",
    "user": "postgres",
    "password": "5657",  # use the password you set in Docker
    "host": "localhost",
    "port": 5432
}

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def get_db_conn():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        st.error("Database connection failed. Please check your DB settings.")
        return None

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import custom modules
from modules.ocr_processor import OCRProcessor
from modules.data_analyzer import DataAnalyzer
from modules.chatbot import DataChatbot, store_processed_doc_in_rag
from modules.form_utils import FormUtils
# Page configuration
st.set_page_config(
    page_title="📊 form processing data tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main title
st.title("📊 form processing data tool")
st.markdown("**Extract insights from forms and documents, then chat with your data for better team and client communication.**")

# Sidebar for data management (API keys now handled via secrets)
with st.sidebar:
    st.header("📊 Application Status")
    
    # Check API keys from secrets
    try:
        mistral_key = st.secrets["MISTRAL_API_KEY"]
        groq_key = st.secrets["GROQ_API_KEY"]
        st.success("🔐 API Keys: Configured")
        st.info("Ready to process documents!")
    except KeyError as e:
        st.error(f"🔐 Missing API Key: {str(e)}")
        st.error("Please configure secrets in Streamlit Cloud or local secrets.toml file")
        st.stop()
    except Exception as e:
        st.error(f"🔐 Configuration Error: {str(e)}")
        st.stop()
    
    st.divider()
    
    # Data Management
    st.header("📁 Data Management")
    if st.session_state.processed_data:
        st.metric("Processed Documents", len(st.session_state.processed_data))
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.processed_data = []
            st.session_state.chat_history = []
            clear_documents_table()
            st.success("All data cleared!")
            st.rerun()
    else:
        st.info("No documents processed yet")

# Initialize processors with API keys from secrets
try:
    ocr_processor = OCRProcessor(mistral_key)
    data_analyzer = DataAnalyzer(groq_key)
    chatbot = DataChatbot(groq_key)
    form_utils = FormUtils()
except Exception as e:
    st.error(f"Error initializing processors: {str(e)}")
    st.error("Please check your API keys configuration in secrets")
    st.stop()

# Main interface tabs
tab1, tab2, tab3 = st.tabs(["📤 Document Processing", "💬 Data Chatbot", "📈 Analytics Dashboard"])

with tab1:
    st.header("📤 Document Upload & Processing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your form/document (image, PDF, DOCX, or text)",
            type=["jpg", "jpeg", "png", "pdf", "txt", "docx"],
            help="Supported formats: JPG, PNG, PDF, DOCX, TXT"
        )
        extracted_text = None
        file_type = None
        if uploaded_file:
            if uploaded_file.type.startswith('image/'):
                file_type = 'image'
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Document", use_container_width=True)
            elif uploaded_file.type == 'application/pdf':
                file_type = 'pdf'
                try:
                    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    extracted_text = "\n".join(page.get_text() for page in pdf_doc)
                    st.info(f"Extracted text from PDF ({uploaded_file.name})")
                except Exception as e:
                    logging.error(f"PDF extraction failed: {e}")
                    st.error("Failed to extract text from PDF. Please check your file.")
            elif uploaded_file.type == 'text/plain':
                file_type = 'text'
                try:
                    extracted_text = uploaded_file.read().decode("utf-8")
                except Exception as e:
                    logging.error(f"Text file extraction failed: {e}")
                    st.error("Failed to read text file.")
            elif uploaded_file.type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword'] or uploaded_file.name.endswith('.docx'):
                file_type = 'docx'
                try:
                    doc = docx.Document(io.BytesIO(uploaded_file.read()))
                    extracted_text = "\n".join([para.text for para in doc.paragraphs])
                    st.info(f"Extracted text from DOCX ({uploaded_file.name})")
                except Exception as e:
                    logging.error(f"DOCX extraction failed: {e}")
                    st.error("Failed to extract text from DOCX. Please check your file.")
            else:
                st.error("Unsupported file type. Please upload an image, PDF, DOCX, or text file.")
                logging.warning(f"Unsupported file type uploaded: {uploaded_file.type}")
            
            # Process button
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        if file_type == 'image':
                            st.info("🔍 Extracting text with Mistral OCR...")
                            ocr_result = ocr_processor.process_image(uploaded_file)
                            if not ocr_result or not ocr_result.get('text'):
                                st.error("❌ Failed to extract text. Please check your image quality.")
                                if ocr_result and ocr_result.get('error'):
                                    st.error(f"Error details: {ocr_result['error']}")
                                logging.error(f"OCR extraction failed for image: {uploaded_file.name}")
                                st.stop()
                            extracted_text = ocr_result['text']
                        elif file_type in ['pdf', 'text', 'docx']:
                            if not extracted_text:
                                st.error("No text could be extracted from the file.")
                                logging.error(f"No text extracted from {file_type} file: {uploaded_file.name}")
                                st.stop()
                            ocr_result = {'text': extracted_text, 'form_type': 'general'}
                        else:
                            st.error("Unsupported file type for processing.")
                            logging.error(f"Unsupported file type for processing: {uploaded_file.type}")
                            st.stop()
                        # Form analysis
                        st.info("🧠 Analyzing document with AI...")
                        analysis_result = data_analyzer.analyze_document(
                            ocr_result['text'], 
                            ocr_result['form_type']
                        )
                        
                        # Combine results
                        processed_doc = {
                            'filename': uploaded_file.name,
                            'timestamp': form_utils.get_timestamp(),
                            'ocr_result': ocr_result,
                            'analysis': analysis_result,
                            'id': len(st.session_state.processed_data) + 1
                        }
                        
                        # Validate document data
                        is_valid, validation_message = form_utils.validate_document_data(processed_doc)
                        if not is_valid:
                            st.error(f"Document validation failed: {validation_message}")
                            logging.error(f"Document validation failed: {validation_message}")
                            st.stop()
                        
                        # Store in session state
                        st.session_state.processed_data.append(processed_doc)
                        store_processed_doc_in_rag(processed_doc)
                        
                        st.success("✅ Document processed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        logging.error(f"Processing failed: {e}")
                        st.exception(e)  # Show full traceback in development
    
    with col2:
        # Results display
        if st.session_state.processed_data:
            st.subheader("📋 Latest Results")
            latest_doc = st.session_state.processed_data[-1]
            
            # Document info
            st.info(f"**File**: {latest_doc['filename']}")
            st.info(f"**Type**: {latest_doc['ocr_result']['form_type'].title()}")
            st.info(f"**Processed**: {latest_doc['timestamp']}")
            
            # Tabbed results
            result_tab1, result_tab2, result_tab3 = st.tabs(["📄 Text", "🔑 Key-Values", "📝 Summary"])
            
            with result_tab1:
                st.text_area(
                    "Extracted Text",
                    latest_doc['ocr_result']['text'],
                    height=200,
                    key="ocr_text_display"
                )
                
                # Download button
                st.download_button(
                    "📥 Download Text",
                    data=latest_doc['ocr_result']['text'],
                    file_name=f"{latest_doc['filename']}_extracted.txt",
                    mime="text/plain"
                )
            
            with result_tab2:
                st.text_area(
                    "Key-Value Pairs & Completeness",
                    latest_doc['analysis']['key_values'],
                    height=200,
                    key="kv_display"
                )
            
            with result_tab3:
                st.text_area(
                    "AI Summary",
                    latest_doc['analysis']['summary'],
                    height=200,
                    key="summary_display"
                )
        else:
            st.info("👆 Upload and process a document to see results here")

with tab2:
    st.header("💬 Chat with Your Data (RAG-powered)")
    if not st.session_state.processed_data:
        st.info("📤 Please process some documents first to enable the chatbot.")
    else:
        user_question = st.chat_input("Ask me anything about your processed documents...")
        if user_question:
            with st.spinner("🤖 Thinking..."):
                query_embedding = get_embedding(user_question)
                context_chunks = fetch_similar_chunks(query_embedding, top_k=3)
                answer = get_llama_rag_response(user_question, context_chunks)
                cleaned_answer, entities = postprocess(answer)
                st.chat_message("assistant").write(cleaned_answer)
                st.write("Named Entities:", entities)

with tab3:
    st.header("📈 Analytics Dashboard")
    
    if not st.session_state.processed_data:
        st.info("📤 Process some documents to see analytics.")
    else:
        # Get statistics
        stats = form_utils.get_document_stats(st.session_state.processed_data)
        
        # Analytics overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", stats['total_documents'])
        
        with col2:
            st.metric("Document Types", len(stats['document_types']))
        
        with col3:
            st.metric("Total Characters", f"{stats['total_characters']:,}")
        
        # Document type distribution
        st.subheader("📊 Document Type Distribution")
        if stats['document_types']:
            st.bar_chart(stats['document_types'])
        
        # Processing report
        st.subheader("📄 Processing Report")
        report_data = []
        for doc in st.session_state.processed_data:
            report_data.append({
                'Document': doc['filename'],
                'Type': doc['ocr_result']['form_type'],
                'Processed': doc['timestamp'],
                'Characters': len(doc['ocr_result']['text'])
            })
        
        if report_data:
            st.dataframe(report_data, use_container_width=True)
        
        # Export functionality
        st.subheader("📥 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Analytics as JSON"):
                export_data = {
                    'stats': stats,
                    'documents': st.session_state.processed_data,
                    'export_timestamp': form_utils.get_timestamp()
                }
                st.download_button(
                    "📥 Download Analytics Data",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"form_analytics_{form_utils.get_timestamp().replace(':', '-').replace(' ', '_')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📋 Export Processing Summary"):
                summary_text = f"Form Processing Summary - {form_utils.get_timestamp()}\n"
                summary_text += "=" * 50 + "\n\n"
                summary_text += f"Total Documents Processed: {stats['total_documents']}\n"
                summary_text += f"Document Types: {len(stats['document_types'])}\n"
                summary_text += f"Total Characters Extracted: {stats['total_characters']:,}\n\n"
                
                summary_text += "Document Details:\n"
                summary_text += "-" * 20 + "\n"
                for i, doc in enumerate(st.session_state.processed_data):
                    summary_text += f"{i+1}. {doc['filename']}\n"
                    summary_text += f"   Type: {doc['ocr_result']['form_type']}\n"
                    summary_text += f"   Processed: {doc['timestamp']}\n"
                    summary_text += f"   Summary: {doc['analysis']['summary'][:100]}...\n\n"
                
                st.download_button(
                    "📥 Download Summary Report",
                    data=summary_text,
                    file_name=f"processing_summary_{form_utils.get_timestamp().replace(':', '-').replace(' ', '_')}.txt",
                    mime="text/plain"
                )

# Footer
st.divider()
st.markdown("---")
st.markdown("**📊 Form Processing Data Tool** - Powered by Mistral OCR & Groq LLaMA AI")
st.markdown("Process documents seamlessly without managing API keys. Ready for production deployment!")

def insert_chunk(text, embedding):
    conn = get_db_conn()
    if conn is None:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                    (text, embedding)
                )
    except Exception as e:
        logging.error(f"Failed to insert chunk: {e}")
        st.error("Failed to store a document chunk in the database.")
    finally:
        conn.close()

def fetch_similar_chunks(query_embedding, top_k=3):
    conn = get_db_conn()
    if conn is None:
        return []
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content, embedding <#> %s AS distance
                    FROM documents
                    ORDER BY embedding <#> %s
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k)
                )
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logging.error(f"Failed to fetch similar chunks: {e}")
        st.error("Failed to retrieve relevant document chunks from the database.")
        return []
    finally:
        conn.close()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --- CHUNKING ---
def chunk_text(text, chunk_size=300):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# --- EMBEDDING ---
# Remove any openai embedding code, as sentence-transformers is used instead.

# --- POSTPROCESSING ---
def postprocess(text):
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return text, entities

# --- RAG PROMPT ---
def build_prompt(user_query, context_chunks, few_shot_examples=None):
    prompt = ""
    if few_shot_examples:
        prompt += few_shot_examples + "\n"
    prompt += "Context:\n" + "\n".join(context_chunks) + "\n"
    prompt += f"Q: {user_query}\nA:"
    return prompt

# --- After document processing, store full text in RAG vector DB ---
def store_processed_doc_in_rag(doc):
    text_to_chunk = doc['ocr_result']['text']
    chunks = chunk_text(text_to_chunk)
    for chunk in chunks:
        embedding = get_embedding(chunk)
        if embedding is not None:
            insert_chunk(chunk, embedding)
        else:
            logging.warning(f"Skipping chunk due to embedding failure: {chunk[:50]}")

# --- In your document processing/upload logic, after processing each doc ---
# Example (pseudo):
# for doc in processed_docs:
#     store_processed_doc_in_rag(doc)
#     st.session_state.processed_data.append(doc)
