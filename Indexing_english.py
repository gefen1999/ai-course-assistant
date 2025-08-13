import re
import json
import zipfile
from pathlib import Path

import pdfplumber
import docx
from pdfminer.high_level import extract_text

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter
)
from pinecone import Pinecone, ServerlessSpec

import spacy
import subprocess

def ensure_spacy_model(model_name="en_core_web_sm"):
    """Ensure that the given spaCy model is installed."""
    try:
        spacy.load(model_name)
    except OSError:
        print(f"⚠️ spaCy model '{model_name}' not found. Downloading...")
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        print(f"✅ Downloaded spaCy model: {model_name}")

# ─── Chunking Strategies ──────────────────────────────────────────────────

def recursive_chunking(texts: list[str], chunk_size: int, overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

def overlapping_chunking(texts: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    step = chunk_size - overlap
    for text in texts:
        words = text.split()
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk.strip())
    return chunks

def spacy_chunking(texts: list[str], chunk_size: int, overlap: int) -> list[str]:
    splitter = SpacyTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

def select_chunking_strategy(strategy_name: str):
    strategies = {
        "recursive": recursive_chunking,
        "overlapping": overlapping_chunking,
        "spacy": spacy_chunking
    }
    return strategies.get(strategy_name)

# ─── Preprocessing Functions ──────────────────────────────────────────────

def extract_zip_file(zip_path: Path, extract_to: Path):
    if not extract_to.exists():
        extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def remove_cid_artifacts(text):
    return re.sub(r'\(cid:\d+\)', '', text)

def read_pdf(path: Path) -> str:
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"⚠️ Error reading {path.name}: {e}")
    return remove_cid_artifacts(text)

def read_docx(path: Path) -> str:
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\.\,\;\:\?\!\-\s]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def embed_chunks(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype="float32")
    embs = model.encode(texts, convert_to_numpy=True)
    return normalize(embs, axis=1).astype("float32")

# ─── Pinecone Integration ─────────────────────────────────────────────────

def create_index(pc, index_name, dimension):
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created new Pinecone index: {index_name}")
    else:
        print(f"Using existing Pinecone index: {index_name}")

def save_to_pinecone(records, embeddings, pc, index_name, namespace="default"):
    index = pc.Index(index_name)
    vectors = []
    for i, record in enumerate(records):
        vectors.append({
            "id": f"{Path(record['file']).stem}_chunk_{record['chunk_id']}",
            "values": embeddings[i].tolist(),
            "metadata": {
                "file": Path(record['file']).name,
                "chunk_id": record['chunk_id'],
                "text": record['text'][:1000]
            }
        })
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100], namespace=namespace)
    print(f"📌 Saved {len(vectors)} vectors to Pinecone namespace '{namespace}'.")

# ─── Main Function ────────────────────────────────────────────────────────

def main():
    MODEL_NAME = "all-MiniLM-L6-v2"
    ZIP_PATH = Path("Eng_data.zip")
    DATA_DIR = ZIP_PATH.with_suffix("")
    CHUNK_SIZE = 200
    OVERLAP = 100
    PINECONE_INDEX_NAME = "document-search"
    CHUNK_STRATEGY = "spacy"  # Options: recursive / overlapping / spacy
    NAMESPACE = f"ENG-{CHUNK_STRATEGY}"

    with open(r"../src/api_keys.json") as f:
        api_keys = json.load(f)
    pc = Pinecone(api_key=api_keys["pinecone"])

    print(f"[1/5] Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    dimension = model.get_sentence_embedding_dimension()

    create_index(pc, PINECONE_INDEX_NAME, dimension)

    print("[2/5] Extracting zip and reading documents...")
    extract_zip_file(ZIP_PATH, DATA_DIR)
    raw_texts = []
    sources = []

    all_files = list(DATA_DIR.rglob("*"))
    with tqdm(all_files, desc="Reading files") as pbar:
        for path in pbar:
            if path.suffix.lower() == ".pdf":
                text = read_pdf(path)
            elif path.suffix.lower() == ".docx":
                text = read_docx(path)
            else:
                continue
            cleaned = clean_text(text)
            raw_texts.append(cleaned)
            sources.append(str(path))

    print("[3/5] Chunking text...")
    if CHUNK_STRATEGY == "spacy":
      ensure_spacy_model("en_core_web_sm")

    chunk_func = select_chunking_strategy(CHUNK_STRATEGY)
    chunks = chunk_func(raw_texts, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "file": sources[i % len(sources)],
            "chunk_id": i,
            "text": chunk
        })

    print(f"→ Created {len(records)} chunks")

    print("[4/5] Embedding and uploading to Pinecone...")
    embs = embed_chunks([r["text"] for r in records], model)
    save_to_pinecone(records, embs, pc, PINECONE_INDEX_NAME, namespace=NAMESPACE)

    print(f"\n✅ Done! Uploaded {len(records)} chunks to Pinecone namespace '{NAMESPACE}'.")

if __name__ == "__main__":
    main()
