import streamlit as st
import pdfplumber
import torch
import torch.nn.functional as F
import numpy as np
import pickle

from transformers import BertForSequenceClassification, BertTokenizer

# -------------------------
# Load Model + Tokenizer
# -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "resume_model"     # your saved folder

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
final_model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

# -------------------------
# Load Label Encoder
# -------------------------
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# -------------------------
# PDF Extraction
# -------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text


# -------------------------
# Simple Preprocessing
# -------------------------
def preprocess(text):
    import re
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------
# Softmax Role Prediction
# -------------------------
def get_top5_softmax(text, max_length=256):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = final_model(**enc).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    categories = label_encoder.inverse_transform(range(len(probs)))
    ranked = sorted(zip(categories, probs), key=lambda x: x[1], reverse=True)
    return ranked[:5]


# -------------------------
# BERT Embeddings (CLS token)
# -------------------------
def get_embedding(text, max_length=256):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = final_model.bert(**enc)
        # Prefer pooled output; fall back to CLS token if not available
        if outputs.pooler_output is not None:
            emb = outputs.pooler_output
        else:
            emb = outputs.last_hidden_state[:, 0, :]

    return emb.cpu().squeeze(0)


# -------------------------
# Cosine Similarity
# -------------------------
def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# -------------------------
# Streamlit UI
# -------------------------
st.title("📄 Resume Classification & Similarity Checker (Fine-Tuned BERT)")
st.write("Upload a resume and optionally paste a job description to analyze alignment.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description (optional):")

if resume_file:
    # Extract and preprocess
    raw_text = extract_text_from_pdf(resume_file)
    clean_text = preprocess(raw_text)

    st.subheader("📄 Extracted Resume Text (Preview)")
    st.write(clean_text[:800] + "...")

    # -------------------------
    # Softmax Predictions
    # -------------------------
    st.subheader("🔍 Top 5 Predicted Job Roles (Softmax Confidence)")
    top5 = get_top5_softmax(clean_text)

    for role, score in top5:
        st.write(f"**{role}** — {score:.4f}")

    # -------------------------
    # Cosine Similarity to JD
    # -------------------------
    if jd_text.strip():
        st.subheader("📊 Resume ↔ Job Description Cosine Similarity")

        jd_clean = preprocess(jd_text)

        resume_emb = get_embedding(clean_text)
        jd_emb = get_embedding(jd_clean)

        sim_score = cosine_similarity(resume_emb, jd_emb)

        st.write(f"**Similarity Score:** {sim_score:.4f}")

