import psycopg2
import spacy
import re
from groq import Groq
from modules.rag_utils import chunk_text, get_embedding, postprocess, insert_chunk, fetch_similar_chunks, store_processed_doc_in_rag
import streamlit as st
import logging

# RAG prompt builder
def build_rag_prompt(user_query, context_chunks):
    """Build RAG prompt with context and query"""
    context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
    prompt = (
        f"Based on the following context from processed documents, answer the user's question accurately.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Use only the provided context to answer.\n"
        f"- If the answer is not in the context, say 'I don't know.'\n"
    )
    # Debug: Print context and prompt
    print("\n--- RAG Context Chunks ---")
    for i, chunk in enumerate(context_chunks):
        print(f"Chunk {i+1}: {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
    print("\n--- RAG Prompt Sent to Llama ---\n" + prompt)
    return prompt

# Llama 3-70B via Groq for RAG chatbot response
def get_llama_rag_response(user_query, context_chunks):
    """Get response from Llama model using RAG context"""
    try:
        groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        prompt = build_rag_prompt(user_query, context_chunks)
        
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Llama RAG response failed: {e}")
        return f"I encountered an error while processing your question: {str(e)}. Please try again."

class DataChatbot:
    """Enhanced chatbot for interacting with processed document data."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Groq(api_key=api_key)

    def prepare_context(self, processed_data):
        """Prepare context from processed documents"""
        context = "Available Documents and Data:\n\n"
        for i, doc in enumerate(processed_data, 1):
            context += f"Document {i}: {doc['filename']}\n"
            context += f"- Type: {doc['ocr_result'].get('form_type', 'unknown')}\n"
            context += f"- Processed: {doc['timestamp']}\n"
            context += f"- Text Preview: {doc['ocr_result'].get('text', '')[:200]}...\n"
            
            if 'analysis' in doc and 'summary' in doc['analysis']:
                context += f"- Summary: {doc['analysis']['summary'][:150]}...\n"
            
            if 'structured_data' in doc:
                context += "- Extracted Key Fields:\n"
                for k, v in doc['structured_data'].items():
                    context += f"   - {k}: {v}\n"
            
            context += "\n" + "="*50 + "\n\n"
        return context

    def generate_response(self, user_question: str, context: str) -> str:
        """Generate response using Groq Llama model"""
        try:
            system_prompt = (
                "You are a helpful AI assistant that provides accurate insights from document processing.\n"
                "Your responsibilities:\n"
                "- Accurately answer user questions using ONLY the structured data and summaries provided.\n"
                "- If the required data isn't in the context, explain that clearly and suggest next steps.\n"
                "- Do NOT make up values. Focus on practical business insights from ANY type of form (tax, medical, insurance, college, employment, etc).\n"
                "- Be concise, factual, and helpful for team communication and decision-making.\n"
                "- When referencing specific documents, mention their filenames for clarity."
            )
            
            user_prompt = f"""
Below is the available context from processed documents:

{context}

Now answer this question clearly and factually:

{user_question}
"""
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=1000
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Response generation failed: {e}")
            return f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or check if the API service is available."

    def chat_with_data(self, user_question: str, processed_data):
        """Main chat interface with processed data"""
        if not processed_data:
            return "I don't have any processed documents to analyze yet. Please upload and process some documents first."
        
        context = self.prepare_context(processed_data)
        return self.generate_response(user_question, context)

    def rag_chat(self, user_question: str):
        """RAG-based chat using vector similarity search"""
        try:
            # Generate embedding for the query
            query_embedding = get_embedding(user_question)
            if query_embedding is None:
                return "I couldn't process your question. Please try again."
            
            # Fetch similar chunks
            context_chunks = fetch_similar_chunks(query_embedding, top_k=3)
            
            if not context_chunks:
                return "I couldn't find relevant information in the processed documents. Please make sure you've uploaded and processed some documents first."
            
            # Generate response using RAG
            response = get_llama_rag_response(user_question, context_chunks)
            return response
            
        except Exception as e:
            logging.error(f"RAG chat failed: {e}")
            return f"I encountered an error while searching for relevant information: {str(e)}. Please try again."