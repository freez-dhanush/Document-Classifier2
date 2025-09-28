import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PyPDF2 import PdfReader
import os

# --- CONFIGURATION (CORRECTED) ---
# Use a relative path to the model folder
MODEL_DIR = "roberta-doc-classifier" 

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            # Add a space to prevent words from merging between pages
            text += page.extract_text() + " " 
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

@st.cache_resource
def load_model():
    """Loads the tokenizer and model from the local directory."""
    if not os.path.isdir(MODEL_DIR):
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None, None

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("📄 State-of-the-Art Document Classifier")
st.markdown("Powered by **RoBERTa**. Upload a PDF document (resume, invoice, or scientific paper) to see it in action.")

tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error(f"Model not found. Please make sure the '{MODEL_DIR}' folder is in the same directory as this app.")
else:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.info("File uploaded successfully. Analyzing...")

        document_text = extract_text_from_pdf(uploaded_file)

        if document_text:
            with st.spinner('Model is predicting...'):
                # 1. Tokenize the extracted text
                inputs = tokenizer(document_text, padding="max_length", truncation=True, return_tensors="pt")

                # 2. Get model prediction (logits)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                # 3. Convert logits to probabilities and get the predicted class
                probabilities = torch.softmax(logits, dim=1).squeeze()
                predicted_class_id = torch.argmax(probabilities).item()
                predicted_class_name = model.config.id2label[predicted_class_id]
                confidence_score = probabilities[predicted_class_id].item()

                # Display the results
                st.success(f"**Prediction Complete!**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Document Type", f"**{predicted_class_name.upper()}**")
                with col2:
                    st.metric("Confidence Score", f"**{confidence_score:.2%}**")

                with st.expander("Show Extracted Text"):
                    st.text_area("", document_text, height=300)