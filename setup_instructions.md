# Form Processing System - Setup Instructions

## For Windows Local Development

### Quick Start Commands

1. **Navigate to your project directory:**
```cmd
cd C:\Users\hp\Desktop\mistralocrextraction\complete_form_procesing\ProductDataPro
```

2. **Activate your virtual environment:**
```cmd
formm\Scripts\activate
```

3. **Install dependencies:**
```cmd
pip install streamlit groq mistralai pillow pydantic python-dotenv
```

4. **Run the application:**
```cmd
streamlit run app.py --server.port 8501
```

**Alternative method:**
```cmd
python run_local.py
```

### Directory Structure Required
```
ProductDataPro/
├── app.py                 # Main application
├── run_local.py          # Local runner script
├── project_requirements.txt
├── __init__.py
├── modules/
│   ├── __init__.py
│   ├── ocr_processor.py
│   ├── data_analyzer.py
│   ├── chatbot.py
│   └── form_utils.py
└── .streamlit/
    └── config.toml
```

### API Keys Setup

Set environment variables or enter in the app sidebar:
- `MISTRAL_API_KEY` - For OCR text extraction
- `GROQ_API_KEY` - For AI analysis and chatbot

### Troubleshooting

If you get "Relative module names not supported":
1. Ensure all `__init__.py` files exist
2. Use the `run_local.py` script instead
3. Check that you're in the correct directory

### Usage
1. Enter API keys in sidebar
2. Upload medical/insurance/financial forms (JPG, PNG, PDF)
3. Click "Process Document" to extract and analyze
4. Use the chatbot to interact with processed data
5. Export data from Analytics Dashboard