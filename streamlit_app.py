import streamlit as st
from rag_utils import load_api_keys, load_model, embed_query, search_pinecone, get_pinecone_client
from chunking import select_chunking_strategy
import os
import openai
import uuid
from openai import AzureOpenAI
from pinecone import Pinecone
import pathlib
import time

# API keys and configuration
with open(r"../src/api_keys.json") as f:
    api_keys = json.load(f)

params = ['endpoint', 'model_name', 'deployment', 'subscription_key', 'api_version']
for p in params:
    if p not in api_keys:
        raise ValueError(f"Missing '{p}' in api_keys.json")

endpoint = api_keys["endpoint"]
model_name = api_keys["model_name"]
deployment = api_keys["deployment"]
subscription_key = api_keys["subscription_key"]
api_version = api_keys["api_version"]

st.set_page_config(page_title="AI Course Assistant", layout="wide")

# Custom CSS for Technion style
st.markdown("""
    <style>
    body {
        background-color: #fff !important;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .stButton>button {
        background-color: #0033a0;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }
    .stTextArea textarea {
        background-color: #f5faff;
        border-radius: 8px;
    }
    .stMarkdown {
        color: #0033a0;
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
with col1:
    st.image("faculty.png", width=150)
with col3:
    st.image("technion_logo.png", width=150)

st.title("AI Course Assistant")
st.markdown(
    "<span style='font-size:20px;'>"
    "🔍 Welcome to the Data and Decision Sciences Faculty AI Course Assistant!<br>"
    "Search and explore your faculty's OneDrive documents with smart AI-powered answers."
    "</span>",
    unsafe_allow_html=True
)
MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
PINECONE_INDEX_NAME = "dot-product"
CHUNKING_OPTIONS = {
    "Recursive": "recursive",
    "Overlapping": "overlapping",
    "Spacy": "spacy"
}

CHUNK_SIZE = 1500
OVERLAP = 200

def get_courses_from_folder(folder_path="Eng_data"):
    folder = pathlib.Path(folder_path)
    files = [f.stem for f in folder.glob("*") if f.is_file()]
    return sorted(files)

def generate_answer_azure(prompt, context, course, azure_endpoint, azure_key, deployment_name):
    client = openai.AzureOpenAI(
        api_key=azure_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint
    )
    course_prompt = ''
    if course!= 'Search in all documents':
        course_prompt = f'Specifically, about {course} course taught at the Technion.'
    messages = [
        {
            "role": "system",
            "content": (
                "You will receive:\n"
                "- A user question about study material.\n"
                f"{course_prompt if course else ''}\n"
                "- Up to three context chunks related to the question.\n\n"
                "Instructions:\n"
                "1. Carefully read the question.\n"
                "2. Use your own knowledge to answer.\n"
                "3. Refer to the context chunks for supporting details or corrections.\n"
                "4. If the context does not help answer the question, say so."
                "5. Summarize the provided context at the end."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {prompt}"
        }]

    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        max_tokens=512,
        temperature=0.2
    )
    return response.choices[0].message.content

@st.cache_resource
def setup():
    api_keys = load_api_keys()
    model = load_model(MODEL_NAME)
    pc = get_pinecone_client(api_keys["pinecone_anna"])
    return model, pc

model, pc = setup()

with st.sidebar:
    st.header("Settings")
    courses = ['Search in all documents']
    courses.extend(get_courses_from_folder())
    selected_course = st.selectbox("Choose a course:", courses)
    chunking_choice = st.selectbox("Choose chunking method:", list(CHUNKING_OPTIONS.keys()))
    chunking_strategy = select_chunking_strategy(CHUNKING_OPTIONS[chunking_choice])
    namespace = f"ENG-{CHUNKING_OPTIONS[chunking_choice]}"
    st.markdown("---")
    st.header("Add Your Material")
    if "show_text_input" not in st.session_state:
        st.session_state.show_text_input = False
    if st.button("I want to add my own material", key="sidebar_add_material"):
        st.session_state.show_text_input = not st.session_state.show_text_input
    text_input = ""
    if st.session_state.show_text_input:
        text_input = st.text_area("Paste your document text here:", key="sidebar_text_area")
        if st.button("Chunk Text", key="sidebar_chunk_text"):
            if text_input:
                chunks = chunking_strategy([text_input], CHUNK_SIZE, OVERLAP)
                st.write(f"Generated {len(chunks)} chunks:")
                st.write("Uploading chunks to Pinecone...")
                vectors = []
                for i, chunk in enumerate(chunks):
                    emb = embed_query(chunk, model)
                    vector_id = str(uuid.uuid4())
                    metadata = {"text": chunk, "file": "user_input"}
                    vectors.append({"id": vector_id, "values": emb.tolist(), "metadata": metadata})
                index = pc.Index(PINECONE_INDEX_NAME)
                index.upsert(vectors=vectors, namespace=namespace)
                st.success(f"Uploaded {len(chunks)} chunks to Pinecone.")
            else:
                st.warning("Please enter some text to chunk.")



# Single chat history for all courses
if "chat_history_all" not in st.session_state:
    st.session_state["chat_history_all"] = []

# Track previous selected course
if "prev_selected_course" not in st.session_state:
    st.session_state.prev_selected_course = selected_course

# Detect course change and add system message
if selected_course != st.session_state.prev_selected_course:
    st.session_state["chat_history_all"].append({
        "role": "system",
        "content": f"Course changed to {selected_course}."
    })
    st.session_state.prev_selected_course = selected_course
    st.session_state["show_course_warning"] = True
    st.session_state.prev_selected_course = selected_course
else:
    st.session_state["show_course_warning"] = False

# Show warning if course changed (place this before chat history)
if st.session_state.get("show_course_warning", False):
    st.markdown(
        f"<div style='background-color:#fff3cd;border-left:6px solid #ffe066;padding:10px;margin-bottom:10px;'>"
        f"<strong>Notice:</strong> Course changed to <b>{selected_course}</b>."
        f"</div>",
        unsafe_allow_html=True
    )
if "course_warning_time" in st.session_state:
    elapsed = time.time() - st.session_state["course_warning_time"]
    if elapsed < 5:
        st.markdown(
            f"<div style='background-color:#fff3cd;border-left:6px solid #ffe066;padding:10px;margin-bottom:10px;'>"
            f"<strong>Notice:</strong> Course changed to <b>{selected_course}</b>."
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        del st.session_state["course_warning_time"]
        st.rerun()


# Display chat history
for msg in st.session_state["chat_history_all"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input(f"Ask about {selected_course}...")
if prompt:
    st.session_state["chat_history_all"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("AI is thinking..."):
        query_emb = embed_query(prompt, model)
        filter = None
        if selected_course != 'Search in all documents':
            relevant = [selected_course]
            if text_input:
                relevant.append('user_input')
            filter = {"file": {"$in": relevant}}
        results = search_pinecone(query_emb, pc, PINECONE_INDEX_NAME, namespace, top_k=3, filter=filter)
        context = "\n\n".join([r["metadata"]["text"] for r in results]) if results else ''
        answer = generate_answer_azure(
            prompt,
            context,
            selected_course,
            azure_endpoint=endpoint,
            azure_key=subscription_key,
            deployment_name=deployment)
    st.session_state["chat_history_all"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar="technion_logo.png"):
        st.markdown(answer)