# rag_utils.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

def load_api_keys(path="../src/api_keys.json"):
    with open(path) as f:
        return json.load(f)

def load_model(model_name="multi-qa-mpnet-base-dot-v1"):
    return SentenceTransformer(model_name)

def embed_query(query, model):
    return model.encode([query])[0]

def search_pinecone(query_emb, pc, index_name, namespace, top_k=5, filter = None):
    print(filter)
    index = pc.Index(index_name)
    res = index.query(vector=query_emb.tolist(), top_k=top_k, namespace=namespace, include_metadata=True, filter=filter)
    print('Search results:', res)
    return res['matches']

def get_pinecone_client(api_key):
    return Pinecone(api_key=api_key)