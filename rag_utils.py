# rag_utils.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import re
from pdfminer.high_level import extract_text
import docx
from sklearn.preprocessing import normalize


def load_api_keys(path="src/api_keys.json"):
    with open(path) as f:
        return json.load(f)

def load_model(model_name="multi-qa-mpnet-base-dot-v1"):
    return SentenceTransformer(model_name)

def embed_query(query, model):
    emb = model.encode([query], convert_to_numpy=True)
    return normalize(emb, axis=1).astype("float32")[0]

def search_pinecone(query_emb, pc, index_name, namespace, top_k=5, filter = None):
    print(filter)
    index = pc.Index(index_name)
    res = index.query(vector=query_emb.tolist(), top_k=top_k, namespace=namespace, include_metadata=True, filter=filter)
    print('Search results:', res)
    return res['matches']

def get_pinecone_client(api_key):
    return Pinecone(api_key=api_key)

def remove_cid_artifacts(text):
    return re.sub(r'\(cid:\d+\)', '', text)

def read_pdf(source) -> str:
    try:
        if hasattr(source, "read"):  # file-like object
            source.seek(0)
            text = extract_text(source)
        else:  # assume path or string
            text = extract_text(str(source))
    except Exception as e:
        print(f"⚠️ Error reading PDF: {e}")
        text = ""
    return remove_cid_artifacts(text)

def read_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\.\,\;\:\?\!\-\s]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()