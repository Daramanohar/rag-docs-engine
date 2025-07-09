"""
Form utilities for data processing, validation, and export functionality
"""
import json
import csv
from datetime import datetime
from io import StringIO
import pandas as pd
import re
import openai

class FormUtils:
    """Utility functions for form processing and data management"""
    
    def __init__(self):
        """Initialize form utilities"""
        pass
    
    def get_timestamp(self):
        """Get current timestamp in readable format"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_iso_timestamp(self):
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()
    
    def validate_document_data(self, doc_data):
        """Validate document data structure"""
        required_fields = ['filename', 'timestamp', 'ocr_result', 'analysis']
        
        if not isinstance(doc_data, dict):
            return False, "Document data must be a dictionary"
        
        for field in required_fields:
            if field not in doc_data:
                return False, f"Missing required field: {field}"
        
        # Validate OCR result structure
        if not isinstance(doc_data['ocr_result'], dict):
            return False, "OCR result must be a dictionary"
        
        ocr_required = ['text', 'form_type']
        for field in ocr_required:
            if field not in doc_data['ocr_result']:
                return False, f"Missing OCR field: {field}"
        
        # Validate analysis structure
        if not isinstance(doc_data['analysis'], dict):
            return False, "Analysis must be a dictionary"
        
        analysis_required = ['summary', 'key_values']
        for field in analysis_required:
            if field not in doc_data['analysis']:
                return False, f"Missing analysis field: {field}"
        
        return True, "Document data is valid"
    
    def export_data(self, processed_data, chat_history=None):
        """Export processed data and chat history to JSON format"""
        export_data = {
            'export_info': {
                'timestamp': self.get_iso_timestamp(),
                'version': '1.0',
                'document_count': len(processed_data),
                'chat_message_count': len(chat_history) if chat_history else 0
            },
            'documents': processed_data,
            'chat_history': chat_history or []
        }
        
        return export_data
    
    def export_to_csv(self, processed_data):
        """Export document data to CSV format"""
        if not processed_data:
            return ""
        
        # Prepare data for CSV
        csv_data = []
        for doc in processed_data:
            row = {
                'Filename': doc['filename'],
                'Document_Type': doc['ocr_result']['form_type'],
                'Processing_Timestamp': doc['timestamp'],
                'Text_Length': len(doc['ocr_result']['text']),
                'OCR_Confidence': doc['ocr_result'].get('confidence', 'N/A'),
                'Extracted_Text': doc['ocr_result']['text'][:500] + '...' if len(doc['ocr_result']['text']) > 500 else doc['ocr_result']['text'],
                'Summary': doc['analysis']['summary'][:300] + '...' if len(doc['analysis']['summary']) > 300 else doc['analysis']['summary'],
                'Key_Values': doc['analysis']['key_values'][:300] + '...' if len(doc['analysis']['key_values']) > 300 else doc['analysis']['key_values']
            }
            csv_data.append(row)
        
        # Convert to CSV string
        output = StringIO()
        if csv_data:
            writer = csv.DictWriter(output, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        
        return output.getvalue()
    
    def get_document_stats(self, processed_data):
        """Get statistics about processed documents"""
        if not processed_data:
            return {
                'total_documents': 0,
                'document_types': {},
                'total_characters': 0,
                'average_confidence': 0,
                'processing_dates': []
            }
        
        # Calculate statistics
        total_docs = len(processed_data)
        doc_types = {}
        total_chars = 0
        confidences = []
        dates = []
        
        for doc in processed_data:
            # Document types
            doc_type = doc['ocr_result']['form_type']
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            # Characters
            total_chars += len(doc['ocr_result']['text'])
            
            # Confidence
            if 'confidence' in doc['ocr_result']:
                confidences.append(doc['ocr_result']['confidence'])
            
            # Dates
            dates.append(doc['timestamp'])
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'total_documents': total_docs,
            'document_types': doc_types,
            'total_characters': total_chars,
            'average_characters_per_doc': total_chars / total_docs if total_docs > 0 else 0,
            'average_confidence': round(avg_confidence, 2),
            'processing_dates': sorted(dates),
            'most_common_type': max(doc_types.items(), key=lambda x: x[1])[0] if doc_types else 'None'
        }
    
    def search_documents(self, processed_data, search_term):
        """Search through processed documents for a specific term"""
        if not processed_data or not search_term:
            return []
        
        search_term_lower = search_term.lower()
        matching_docs = []
        
        for doc in processed_data:
            # Search in text content
            if search_term_lower in doc['ocr_result']['text'].lower():
                matching_docs.append({
                    'document': doc,
                    'match_type': 'content',
                    'relevance': doc['ocr_result']['text'].lower().count(search_term_lower)
                })
            # Search in filename
            elif search_term_lower in doc['filename'].lower():
                matching_docs.append({
                    'document': doc,
                    'match_type': 'filename',
                    'relevance': 1
                })
            # Search in analysis
            elif search_term_lower in doc['analysis']['summary'].lower() or search_term_lower in doc['analysis']['key_values'].lower():
                matching_docs.append({
                    'document': doc,
                    'match_type': 'analysis',
                    'relevance': 1
                })
        
        # Sort by relevance
        matching_docs.sort(key=lambda x: x['relevance'], reverse=True)
        return matching_docs
    
    def format_document_preview(self, doc_data, max_length=200):
        """Format a document for preview display"""
        if not doc_data:
            return "No document data available"
        
        preview = f"""
**{doc_data['filename']}**
Type: {doc_data['ocr_result']['form_type'].title()}
Processed: {doc_data['timestamp']}

Text Preview:
{doc_data['ocr_result']['text'][:max_length]}{'...' if len(doc_data['ocr_result']['text']) > max_length else ''}
"""
        return preview.strip()
    
    def generate_processing_report(self, processed_data):
        """Generate a comprehensive processing report"""
        if not processed_data:
            return "No documents have been processed yet."
        
        stats = self.get_document_stats(processed_data)
        
        report = f"""
# Document Processing Report
Generated: {self.get_timestamp()}

## Summary Statistics
- **Total Documents Processed**: {stats['total_documents']}
- **Total Characters Extracted**: {stats['total_characters']:,}
- **Average Characters per Document**: {stats['average_characters_per_doc']:.0f}
- **Average OCR Confidence**: {stats['average_confidence']}%

## Document Types Distribution
"""
        
        for doc_type, count in stats['document_types'].items():
            percentage = (count / stats['total_documents']) * 100
            report += f"- **{doc_type.title()}**: {count} documents ({percentage:.1f}%)\n"
        
        report += f"\n## Most Common Document Type\n{stats['most_common_type'].title()}"
        
        report += f"\n\n## Processing Timeline\n"
        report += f"First Document: {stats['processing_dates'][0] if stats['processing_dates'] else 'N/A'}\n"
        report += f"Latest Document: {stats['processing_dates'][-1] if stats['processing_dates'] else 'N/A'}"
        
        return report
    
    def clean_text(self, text):
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Remove special characters that might cause issues
        cleaned = cleaned.replace('\x00', '').replace('\ufffd', '')
        
        return cleaned.strip()
    
    def extract_contact_info(self, text):
        """Extract contact information from text"""
        contact_info = {
            'emails': [],
            'phones': [],
            'addresses': []
        }
        
        if not text:
            return contact_info
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        contact_info['emails'] = re.findall(email_pattern, text)
        
        # Phone pattern (various formats)
        phone_pattern = r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10})'
        contact_info['phones'] = re.findall(phone_pattern, text)
        
        return contact_info
