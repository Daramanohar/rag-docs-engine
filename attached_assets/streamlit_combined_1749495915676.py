# streamlit_combined.py

import streamlit as st

st.set_page_config(page_title="📄 Unified Form Processor", layout="wide")

from PIL import Image
import os
from Summarization_forms import summarize_text
from KV_and_Completeness_forms import extract_kv_and_check
from ocr_utils import encode_image, process_with_mistral_ocr, convert_to_json, identify_form_type


st.title("📄 Intelligent Form Processor")
st.markdown("Upload a form image. We’ll extract text using **Mistral OCR**, identify form type, and analyze it using **LLaMA 3.3 70B** for Key-Values & Summary.")

# --- Sidebar API keys ---
with st.sidebar:
    st.header("🔐 API Configuration")
    mistral_key = st.text_input("Mistral OCR API Key", type="password")
    groq_key = st.text_input("Groq API Key (for LLaMA)", type="password")
    if not mistral_key or not groq_key:
        st.warning("Please enter both API keys to proceed.")
        st.stop()

# --- File uploader ---
uploaded_file = st.file_uploader("📤 Upload your form image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Form", width=400)

    with st.spinner("Encoding image..."):
        base64_img = encode_image(uploaded_file)
        ext = uploaded_file.name.split(".")[-1]
        if ext == "jpg":
            ext = "jpeg"

    if st.button("🧠 Process Form"):
        with st.spinner("🔍 Running Mistral OCR..."):
            ocr_result = process_with_mistral_ocr(base64_img, ext, mistral_key)
            if not ocr_result:
                st.error("Failed to extract text. Try another image or check API key.")
                st.stop()

            ocr_text = getattr(ocr_result, "text", str(ocr_result))
            form_type = identify_form_type(ocr_text)
            json_data, ocr_text = convert_to_json(ocr_result, ocr_text, form_type)

        tab1, tab2, tab3 = st.tabs(["📄 OCR Text", "🔑 LLaMA KV + Completeness", "📝 LLaMA Summary"])

        with tab1:
            st.subheader(f"Detected Form Type: `{form_type}`")
            st.text_area("Extracted OCR Text", ocr_text, height=300)
            st.download_button("📥 Download Text", data=ocr_text, file_name="form_text.txt")

        with tab2:
            with st.spinner("Running Key-Value & Completeness Extraction..."):
                kv_result = extract_kv_and_check(ocr_text)
                st.text_area("Key-Value Pairs & Completeness", kv_result, height=300)

        with tab3:
            with st.spinner("Generating Summary..."):
                summary = summarize_text(ocr_text, groq_key)
                st.text_area("Form Summary", summary, height=250)
