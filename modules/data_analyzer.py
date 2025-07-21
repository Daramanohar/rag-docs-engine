from groq import Groq
import re
import streamlit as st
from typing import Dict, List, Any

class DataAnalyzer:
    """Handles data analysis using Groq LLaMA API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Groq(api_key=api_key)
    
    def extract_key_values(self, text: str) -> str:
        """
        Extract key-value pairs from text using a robust regex that handles multi-line values.
        """
        try:
            extracted_data = {}

            # This regex is designed to be robust and handle multi-line values.
            # Explanation:
            # ^(?P<key>[A-Za-z0-9\s()\/_-]+?) : Start of a line, capture a non-greedy key containing letters, numbers, spaces, and common symbols.
            # \s*[:|-]\s*                    : Match a separator (colon or pipe) surrounded by optional whitespace.
            # (?P<value>[\s\S]+?)           : Capture a non-greedy value. `[\s\S]` matches ANY character, including newlines.
            # (?=\n^[A-Za-z0-9\s()\/_-]+?\s*[:|-]|\Z) : Positive lookahead. The value ends when we see either:
            #                                       - A newline followed by the start of another key (`\n^...`)
            #                                       - OR the absolute end of the string (`\Z`)
            
            # Note: This is a significant improvement over line-by-line parsing.
            pattern = re.compile(
                r"^(?P<key>[A-Za-z0-9\s()\/_-]+?)\s*[:|-]\s*(?P<value>[\s\S]+?)(?=\n^[A-Za-z0-9\s()\/_-]+?\s*[:|-]|\Z)",
                re.MULTILINE
            )
            
            for match in pattern.finditer(text):
                # Clean up the extracted key and value
                key = match.group('key').strip()
                # The value might have multiple lines, so we just strip leading/trailing whitespace
                value = match.group('value').strip()
                
                if key and key not in extracted_data:
                    extracted_data[key] = value

            # --- Report Generation (Unchanged from original but now uses better data) ---

            # Check for missing fields based on extracted keys
            missing_fields = [key for key, value in extracted_data.items() if not value or value.strip() == ""]

            # Build report
            report = "Extracted Key-Value Pairs:\n"
            report += "=" * 30 + "\n"
            
            if not extracted_data:
                report += "No key-value pairs could be extracted with the defined pattern.\n"
                report += "Please check if the document follows a 'key: value' or 'key | value' format."
            else:
                for k, v in extracted_data.items():
                    status = "✅" if v and v.strip() else "❌"
                    # For display, replace newlines in value with a space to keep the report clean
                    display_value = re.sub(r'\s+', ' ', v) if v else ""
                    report += f"{status} {k}: {display_value}\n"

            report += "\n" + "=" * 30 + "\n"
            
            if extracted_data:
                if missing_fields:
                    completeness = ((len(extracted_data) - len(missing_fields)) / len(extracted_data) * 100)
                    report += f"⚠️ Missing/Empty Fields ({len(missing_fields)}): {', '.join(missing_fields)}\n"
                    report += f"📊 Completeness: {completeness:.1f}%"
                else:
                    report += "🎉 Form appears complete (100%)"

            return report
            
        except Exception as e:
            st.error(f"An unexpected error occurred during key-value extraction: {str(e)}")
            return f"Error extracting key-values: {str(e)}"
    
    def summarize_text(self, text: str, form_type: str = "general") -> str:
        """Generate AI summary of the text."""
        try:
            # Customize prompt based on form type
            system_prompts = {
                "medical": "You are an AI assistant specializing in medical form analysis. Summarize the medical document focusing on patient info, diagnoses, treatments, and key medical details.",
                "insurance": "You are an AI assistant specializing in insurance document analysis. Summarize the insurance form focusing on coverage details, claims, and policy information.",
                "college": "You are an AI assistant specializing in educational document analysis. Summarize the academic document focusing on student information, courses, and educational details.",
                "employment": "You are an AI assistant specializing in employment document analysis. Summarize the employment form focusing on job details, qualifications, and work-related information.",
                "financial": "You are an expert AI assistant specialized in analyzing financial and tax-related documents. "
                            "Review the following form or document and extract key information such as:\n"
                            "- Declarant or Assessee details (e.g., Name, PAN, Status, Residential Status)\n"
                            "- Relevant financial year or assessment year\n"
                            "- Income estimates, tax amounts, and declaration content\n"
                            "- Financial instruments or savings/investments mentioned\n"
                            "- Any claims, exemptions, or no-tax declarations\n\n"
                            "Then, generate a concise summary explaining:\n"
                            "1. The purpose of the form\n"
                            "2. The type of document (e.g., declaration, return, exemption)\n"
                            "3. Key extracted fields and their significance\n"
                            "4. Eligibility or compliance conditions if applicable\n\n"
                            "Keep the tone professional and the summary clear and suitable for a product manager or auditor reviewing the document.",
                
                "general": "You are an AI assistant that summarizes document content concisely and clearly."
            }
            
            system_prompt = system_prompts.get(form_type, system_prompts["general"])
            
            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile", # Note: I've updated to a generally available high-performance model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please provide a comprehensive summary of this {form_type} document:\n\n{text}"}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def analyze_document_completeness(self, text: str, form_type: str) -> Dict[str, Any]:
        """Analyze document completeness using AI."""
        try:
            prompt = f"""
            Analyze this {form_type} document for completeness and important information:

            {text}

            Please provide:
            1. A completeness score (0-100%)
            2. List of missing critical information
            3. Key insights for product managers
            4. Recommendations for team/client communication

            Format your response as a structured analysis.
            """

            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an AI assistant helping product managers analyze documents for team and client communication."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )

            return {
                "ai_analysis": response.choices[0].message.content,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "ai_analysis": f"Error in AI analysis: {str(e)}",
                "status": "error"
            }
    
    def analyze_document(self, text: str, form_type: str) -> Dict[str, Any]:
        """Main method to analyze a document."""
        try:
            # Extract key-value pairs
            key_values = self.extract_key_values(text)
            
            # Generate summary
            summary = self.summarize_text(text, form_type)
            
            # Analyze completeness
            completeness_analysis = self.analyze_document_completeness(text, form_type)
            
            return {
                "key_values": key_values,
                "summary": summary,
                "completeness_analysis": completeness_analysis,
                "form_type": form_type,
                "text_length": len(text),
                "status": "success"
            }
            
        except Exception as e:
            st.error(f"Document analysis failed: {str(e)}")
            return {
                "key_values": f"Error: {str(e)}",
                "summary": f"Error: {str(e)}",
                "completeness_analysis": {"ai_analysis": f"Error: {str(e)}", "status": "error"},
                "form_type": form_type,
                "text_length": 0,
                "status": "error"
            }
