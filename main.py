# main.py

import re
from pathlib import Path

import pdfplumber
import fitz      # PyMuPDF fallback
import docx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss


def read_pdf(path: Path) -> str:
    """Try pdfplumber first, then PyMuPDF if it fails."""
    try:
        text_pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_pages.append(t)
        return "\n\n".join(text_pages)
    except Exception:
        doc = fitz.open(str(path))
        return "\n".join(p.get_text() for p in doc)


def read_docx(path: Path) -> str:
    """Extract all paragraphs from a .docx file."""
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def clean_text(text: str) -> str:
    """Keep Hebrew, Latin, digits, and basic punctuation; collapse whitespace."""
    text = re.sub(r"[^\u0590-\u05FFa-zA-Z0-9\.\,\;\:\?\!\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(
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

    print(f"[1/5] Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    print("[2/5] Reading documents and creating chunks...")
    records = []
    for path in DATA_DIR.rglob("*"):
        if path.suffix.lower() == ".pdf":
            text = read_pdf(path)
        elif path.suffix.lower() == ".docx":
            text = read_docx(path)
        else:
            continue

        chunks = chunk_text(text, MAX_WORDS, OVERLAP, MIN_WORDS)
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
