# main.py

import re
from pathlib import Path
from bidi.algorithm import get_display
import pdfplumber
import fitz      # PyMuPDF fallback
import docx
from pdfminer.high_level import extract_text
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
import zipfile
from tqdm import tqdm


def extract_zips_in_data(data_dir: Path):
    """Extract all .zip files in data_dir into subfolders."""
    for zip_path in data_dir.rglob("*.zip"):
        extract_dir = zip_path.with_suffix("")  # Remove .zip extension
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted {zip_path} to {extract_dir}")


def is_hebrew_or_arabic(char):
    # Hebrew range: \u0590-\u05FF
    # Arabic range: \u0600-\u06FF
    return ('\u0590' <= char <= '\u05FF') or ('\u0600' <= char <= '\u06FF')

def process_line_bidi(line):
    # Apply BiDi algorithm only if line contains Hebrew or Arabic chars
    if any(is_hebrew_or_arabic(ch) for ch in line):
        # For Arabic, you may need reshaping:
        # reshaped_text = arabic_reshaper.reshape(line)
        # return get_display(reshaped_text)
        return get_display(line)
    else:
        return line

def remove_cid_artifacts(text):
    return re.sub(r'\(cid:\d+\)', '', text)

def process_text_bidi(text):
    lines = text.splitlines()
    processed_lines = [process_line_bidi(line) for line in lines]
    return "\n".join(processed_lines)


def read_pdf(path: Path) -> str:
    """Try pdfplumber first, then PyMuPDF if it fails."""
    raw_text = extract_text(path)
    raw_text = remove_cid_artifacts(raw_text)
    logical_text = process_text_bidi(raw_text)
    #print(f"Path: {path}, Logical text: {logical_text}")
    return logical_text

    """try:
        text_pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_pages.append(t)
        print('text_pages', text_pages)
        return "\n\n".join(text_pages)
    except Exception:
        doc = fitz.open(str(path))
        return "\n".join(p.get_text() for p in doc)"""


def read_docx(path: Path) -> str:
    """Extract all paragraphs from a .docx file."""
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def clean_text(text: str) -> str:
    """
    Keep Hebrew, Latin, digits, and basic punctuation; collapse whitespace and blank lines.
    """
    # Remove unwanted characters
    text = re.sub(r"[^\u0590-\u05FFa-zA-Z0-9\.\,\;\:\?\!\-\s]", " ", text)
    # Collapse multiple spaces/tabs into one space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple newlines into a single newline
    text = re.sub(r"\n+", "\n", text)
    # Remove leading/trailing whitespace and blank lines
    return text.strip()


def chunk_text(
    text: str,
    min_words: int = 300,
) -> list[str]:
    """
    Chunk text by paragraphs, each chunk has at least min_words.
    """
    text = clean_text(text)
    paras = [p for p in text.split("\n\n") if p]
    chunks = []
    cur, count = [], 0
    for p in paras:
        words = p.split()
        cur.append(p)
        count += len(words)
        if count >= min_words:
            chunk = "\n\n".join(cur)
            chunks.append(chunk)
            cur, count = [], 0
    # Add any remaining paragraphs as a final chunk if not empty
    if cur:
        chunk = "\n\n".join(cur)
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)
    return chunks

def chunk_text_gefen(
    text: str,
    max_words: int,
    overlap: int,
    min_words: int,
) -> list[str]:
    """
    Split cleaned text into overlapping chunks.
    - max_words: approximate upper bound per chunk
    - overlap: how many words to carry over between chunks
    - min_words: drop any chunk smaller than this
    """
    text = clean_text(text)
    paras = [p for p in text.split("\n\n") if p]

    chunks = []
    cur, count = [], 0
    for p in paras:
        words = p.split()
        if count + len(words) > max_words and cur:
            chunk = " ".join(cur)
            if len(chunk.split()) >= min_words:
                print(f"Appending chunk... {chunk}")
                chunks.append(chunk)
            # carry overlap
            cur = cur[-overlap:]
            count = len(cur)
        cur.extend(words)
        count += len(words)

    # final tail
    if cur:
        chunk = " ".join(cur)
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)

    return chunks


def embed_chunks(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encode + L2-normalize into float32."""
    embs = model.encode(texts, convert_to_numpy=True)
    return normalize(embs, axis=1).astype("float32")


def build_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    """Build an inner-product FAISS index."""
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def save_results(results, output_path, query):
    """Write top-K results to a file with separators."""
    separator = "\n" + "-" * 40 + "\n\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"=== Top {len(results)} results for '{query}' ===\n\n")
        for file, chunk_id, score, snippet in results:
            f.write(f"• File: {file}\n")
            f.write(f"  Chunk #{chunk_id}   Score: {score:.3f}\n")
            f.write(f"  Snippet: {snippet}\n")
            f.write(separator)


def main():
    # ─── Editable parameters ────────────────────────────────────────────────
    MODEL_NAME   = "sentence-transformers/LaBSE"
    DATA_DIR     = Path("data")               # where your PDFs/DOCs live
    OUTPUT_PATH  = "results.txt"
    QUERY        = 'מה זה bfs?'                       # set a default string here if desired
    MAX_WORDS    = 200
    OVERLAP      = 50
    MIN_WORDS    = 50
    TOP_K        = 5
    # ────────────────────────────────────────────────────────────────────────

    # ask user if no fixed QUERY
    if not QUERY:
        QUERY = input("Please enter your search query: ").strip()
    extract_zips_in_data(DATA_DIR)
    print(f"[1/5] Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    print("[2/5] Reading documents and creating chunks...")
    records = []
    all_files = list(DATA_DIR.rglob("*"))
    with tqdm(all_files, desc="Processing files") as pbar:
        for path in pbar:
            pbar.set_postfix_str(str(path))
            if path.suffix.lower() == ".pdf":
                text = read_pdf(path)
            elif path.suffix.lower() == ".docx":
                text = read_docx(path)
            else:
                continue
            chunks = chunk_text(text, 300)
            for i, c in enumerate(chunks):
                records.append({
                    "file": str(path),
                    "chunk_id": i,
                    "snippet": c[:200] + ("…" if len(c) > 200 else ""),
                    "text": c
                })


    df = pd.DataFrame(records)
    print(f"    → {len(df)} chunks created")

    print("[3/5] Embedding chunks and building FAISS index...")
    embs = embed_chunks(df["text"].tolist(), model)
    index = build_index(embs)

    print("[4/5] Searching for query in index...")
    q_emb = normalize(model.encode([QUERY], convert_to_numpy=True), axis=1).astype("float32")
    scores, ids = index.search(q_emb, TOP_K)
    top = []
    for score, idx in zip(scores[0], ids[0]):
        row = df.iloc[idx]
        top.append((row.file, row.chunk_id, float(score), row.snippet))

    print("[5/5] Saving top-K results to file...")
    save_results(top, OUTPUT_PATH, QUERY)
    print(f"\n✅ Done! Saved {len(top)} results to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    main()
