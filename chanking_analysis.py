import os
import json
from pathlib import Path
from pinecone import Pinecone
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Load API Key ===
with open("../src/api_keys.json") as f:
    api_keys = json.load(f)

# === Init Pinecone ===
pc = Pinecone(api_key=api_keys["pinecone"])
index = pc.Index("document-search")

# === Create output directory ===
output_dir = Path("chunk_comparison")
output_dir.mkdir(exist_ok=True)

# === Get index stats ===
stats = index.describe_index_stats()
namespaces = list(stats["namespaces"].keys())
dimension = stats["dimension"]

print(f"🔍 Found {len(namespaces)} namespaces: {namespaces}")

# === 1. Chunk count per namespace ===
chunk_counts = {
    ns: stats["namespaces"][ns]["vector_count"]
    for ns in namespaces
}

plt.figure(figsize=(10, 5))
plt.bar(chunk_counts.keys(), chunk_counts.values(), color="skyblue")
plt.xlabel("Chunking Strategy (Namespace)")
plt.ylabel("Number of Chunks (Vectors)")
plt.title("Chunk Count per Namespace in Pinecone")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "chunk_count_comparison.png", dpi=300)


# === 2. Chunk length distribution (boxplot) ===
chunk_lengths = {ns: [] for ns in namespaces}

for ns in tqdm(namespaces, desc="Fetching chunk texts"):
    try:
        res = index.query(
            vector=[0.0] * dimension,
            top_k=min(1000, chunk_counts[ns]),
            include_metadata=True,
            namespace=ns,
        )
        for match in res["matches"]:
            metadata = match["metadata"]
            text = metadata.get("text", "")
            word_count = len(text.split())
            chunk_lengths[ns].append(word_count)
    except Exception as e:
        print(f"❌ Error in namespace {ns}: {e}")

# Plot boxplot of word counts
plt.figure(figsize=(10, 6))
plt.boxplot(
    [chunk_lengths[ns] for ns in namespaces],
    labels=namespaces,
    patch_artist=True
)
plt.xlabel("Chunking Strategy (Namespace)")
plt.ylabel("Chunk Length (Words)")
plt.title("Word Count Distribution per Namespace (Boxplot)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "chunk_wordcount_boxplot.png", dpi=300)

print("✅ Saved 'chunk_wordcount_boxplot.png' in 'chunk_comparison/'")
