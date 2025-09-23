import zipfile
import io
import pdfplumber
import tiktoken

# Path to your ZIP
zip_path = "raw_data_english.zip"

# Initialize GPT-style tokenizer (Azure GPT-4.1 uses cl100k_base)
enc = tiktoken.get_encoding("cl100k_base")

total_pages = 0
total_tokens_split = 0
total_tokens_gpt = 0

with zipfile.ZipFile(zip_path, "r") as zf:
    for name in zf.namelist():
        if name.lower().endswith(".pdf"):
            print(f"Processing: {name}")
            with zf.open(name) as f:
                pdf_bytes = io.BytesIO(f.read())
                with pdfplumber.open(pdf_bytes) as pdf:
                    total_pages += len(pdf.pages)
                    for page in pdf.pages:
                        text = page.extract_text() or ""
                        # Split-based tokenization
                        tokens_split = text.split()
                        total_tokens_split += len(tokens_split)
                        # GPT-style tokenization (cl100k_base)
                        tokens_gpt = enc.encode(text)
                        total_tokens_gpt += len(tokens_gpt)

print("===================================")
print(f"Total pages across all PDFs: {total_pages}")
print(f"Total tokens (split): {total_tokens_split}")
print(f"Total tokens (GPT-style, Azure GPT-4.1): {total_tokens_gpt}")
