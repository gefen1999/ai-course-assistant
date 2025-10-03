import streamlit as st
from rag_utils import load_api_keys, load_model, embed_query, search_pinecone, get_pinecone_client, read_pdf, read_docx
from chunking import select_chunking_strategy
import os
import openai
import uuid
from openai import AzureOpenAI
from pinecone import Pinecone
import pathlib
import time
import json
import re
import base64
import io

# API keys and configuration
with open(r"src/api_keys.json") as f:
    api_keys = json.load(f)

params = ['endpoint', 'deployment', 'subscription_key', 'api_version']
for p in params:
    if p not in api_keys:
        raise ValueError(f"Missing '{p}' in api_keys.json")

endpoint = api_keys["endpoint"]
#model_name = api_keys["model_name"]
deployment = api_keys["deployment"]
subscription_key = api_keys["subscription_key"]
api_version = api_keys["api_version"]

st.set_page_config(
    page_title="Dasha - AI Academic Assistant",
    page_icon="streamlit_utils/dasha.jpeg",  # Path to your image file
    layout="wide"
)
def display_answer(answer):
    pattern = r"(\$\$.*?\$\$|\$.*?\$)"
    segments = re.split(pattern, answer, flags=re.DOTALL)
    for seg in segments:
        seg = seg.strip()
        if seg.startswith("$$") and seg.endswith("$$"):
            st.latex(seg[2:-2].strip())
        elif seg.startswith("$") and seg.endswith("$"):
            st.latex(seg[1:-1].strip())
        elif seg:
            st.markdown(seg)

# Custom CSS for Technion style
st.markdown("""
    <style>
    body {
        background-color: #fff !important;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .stButton>button {
        background-color: #02468b;
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

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.image("streamlit_utils/HEADER6.png", width=750)  # Adjust width as needed

with col2:
    st.markdown(
        """
        <div style='display: flex; flex-direction: column; justify-content: center; height: 25px;'>
            <div id='how-to-use-btn'></div>
        </div>
        <style>
        [data-testid="column"]:nth-of-type(2) > div {
            height: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("💡 How To Use?", key="how-to-use-btn"):
        st.session_state["show_instructions"] = not st.session_state["show_instructions"]
#st.title("AI Course Assistant")
#st.title("Dasha Chat")
if "show_instructions" not in st.session_state:
    st.session_state["show_instructions"] = False

if st.session_state["show_instructions"]:
    st.markdown(
        "<div style='background-color:#f0f0f0;padding:16px;border-radius:10px;margin-bottom:10px;'>"
        "<span style='font-size:16px;'>"
        "👋 <b> Welcome to Dasha, Data-Driven Academic Smart Helper Agent!</b><br>"
        "To get started:<br>"
        "1. <b>Select a course</b> from the sidebar.<br>"
        "2. <b>Type your question</b> about the course material in the chat box below.<br>"
        "3. Optionally, <b>add your own study material</b> using the sidebar.<br>"
        "Dasha will answer your questions and help you study smarter!"
        "</span>"
        "</div>",
        unsafe_allow_html=True
    )

MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
PINECONE_INDEX_NAME = "dotproduct-300"
CHUNKING_OPTIONS = {
    "Recursive": "recursive",
    "Overlapping": "overlapping",
    "Spacy": "spacy"
}

CHUNK_SIZE = 1000
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
                "You are Dasha, an academic helper agent. You will receive:\n"
                "- A user question about study material.\n"
                f"{course_prompt if course else ''}\n"
                "- Up to three context chunks related to the question.\n\n"
                "Instructions:\n"
                "1. Carefully read the question.\n"
                "2. Use your own knowledge to answer.\n"
                "3. Refer to the context chunks for supporting details or corrections.\n"
                "4. If the provided context does not help answer the question, inform the user and recommend selecting a different course. "
                #"5. Summarize the provided context at the end."
                "5. If the answer contains formulas, use LaTeX delimiters in streamlit format ($...$ or $$...$$)."
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
    pc = get_pinecone_client(api_keys["pinecone"])
    return model, pc

model, pc = setup()
st.markdown("""
    <style>
    .stChatMessageContent {
        font-size: 1.25rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
with st.sidebar:
    st.header("Settings")
    courses = get_courses_from_folder()
    courses.append('Search in all documents')
    selected_course = st.selectbox("Choose a course:", courses)
    chunking_choice = st.selectbox("Choose chunking method:", list(CHUNKING_OPTIONS.keys()))
    chunking_strategy = select_chunking_strategy(CHUNKING_OPTIONS[chunking_choice])
    namespace = f"{CHUNKING_OPTIONS[chunking_choice]}"
    st.markdown("---")
    if "show_text_input" not in st.session_state:
        st.session_state.show_text_input = False

    # 1. Initialize status
    if "upload_status" not in st.session_state:
        st.session_state.upload_status = "idle"

    button_label = "I don't want to add my own material" if st.session_state.show_text_input else "I want to add my own material"
    if st.button(button_label, key="sidebar_add_material"):
        st.session_state.show_text_input = not st.session_state.show_text_input
        st.rerun()  # Force immediate UI update

    text_input = ""
    if st.session_state.show_text_input:
        uploaded_file = st.file_uploader("Upload PDF or DOCX file:", type=["pdf", "docx"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                text_input = read_pdf(io.BytesIO(uploaded_file.read()))
            elif uploaded_file.type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword"
            ]:
                text_input = read_docx(io.BytesIO(uploaded_file.read()))
            else:
                st.warning("Unsupported file type, only .pdf and .docx files are supported.")
            st.text_area("Extracted text:", value=text_input[:2000], height=200, key="sidebar_text_area", disabled=True)
        else:
            text_input = st.text_area("Or paste your document text here:", key="sidebar_text_area")

        # Use a unique key for the chunk button inside the text input block
        if st.button("Upload to Knowledge Base", key="sidebar_chunk_text_inner"):
            if text_input:
                st.session_state.upload_status = "uploading"
                st.rerun()

        if st.session_state.upload_status == "uploading":
            st.markdown(
                "<div style='padding:10px; background:#e6f0fa; border-radius:8px; color:#02468b; font-weight:500;'>"
                "⏳ Uploading chunks to Pinecone..."
                "</div>",
                unsafe_allow_html=True
            )
        elif st.session_state.upload_status == "done":
            st.markdown(
                "<div style='padding:10px; background:#e6f0fa; border-radius:8px; color:#02468b; font-weight:500;'>"
                "✅ Uploaded chunks to Pinecone."
                "</div>",
                unsafe_allow_html=True
            )

        # 4. Actual upload logic (after rerun)
        if st.session_state.upload_status == "uploading" and text_input:
            chunks = chunking_strategy([text_input], CHUNK_SIZE, OVERLAP)
            vectors = []
            for i, chunk in enumerate(chunks):
                emb = embed_query(chunk, model)
                vector_id = str(uuid.uuid4())
                metadata = {"text": chunk, "file": f"{selected_course}_user_input"}
                vectors.append({"id": vector_id, "values": emb.tolist(), "metadata": metadata})
            index = pc.Index(PINECONE_INDEX_NAME)
            index.upsert(vectors=vectors, namespace=namespace)
            st.session_state.upload_status = "done"
            st.rerun()

    st.markdown("---")

    first_message = {
            "role": "assistant",
            "content": (
                "Hey you! I'm Dasha, your AI Academic Assistant. "
                "Ask me anything about your courses or study materials!"
            )
        }
    st.markdown(
        """
        <div style='width:40%; margin:0 auto; display:flex; justify-content:center; align-items:center;'>
            <div id='clear-chat-btn'></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Clear Chat", key="clear_chat_btn"):
        st.session_state["chat_history_all"] = [first_message]
        st.rerun()


# Single chat history for all courses
if "chat_history_all" not in st.session_state:
    st.session_state["chat_history_all"] = []
    # Add initial agent message
    st.session_state["chat_history_all"].append(first_message)

# Track previous selected course
if "prev_selected_course" not in st.session_state:
    st.session_state.prev_selected_course = selected_course
st.markdown("""
    <style>
    .stChatMessage .stChatAvatar img {
        width: 96px !important;
        height: 96px !important;
        object-fit: contain;
    }
    </style>
    """, unsafe_allow_html=True)
# Detect course change and add system message
if selected_course != st.session_state.prev_selected_course:
    st.session_state["chat_history_all"].append({
        "role": "system",
        "content": f"Course changed to {selected_course}."
    })
    st.session_state.prev_selected_course = selected_course
    st.session_state["show_course_warning"] = True
    # Reset user upload state
    st.session_state.upload_status = "idle"
    st.session_state.show_text_input = False

    if "sidebar_text_area" in st.session_state:
        del st.session_state["sidebar_text_area"]
    st.rerun()

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
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="streamlit_utils/dasha.jpeg"):
            st.markdown(msg["content"])
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

st.markdown("</div>", unsafe_allow_html=True)

# Chat input
field_text = f"Ask about {selected_course}..."
if selected_course == 'Search in all documents':
    st.info("Searching across all documents. Responses may be less specific.")
    field_text = f"Ask about any course..."
prompt = st.chat_input(f"Ask about {selected_course}...")
st.markdown(
    """
    <div class='dasha-footer'>
        Dasha is powered by GPT-4.1
    </div>
    <style>
    .dasha-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100vw;
        background: #fff;
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        font-weight: 500;
        z-index: 9999;
        padding: 8px 0 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)
if prompt:
    st.session_state["chat_history_all"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Dasha is thinking..."):
        query_emb = embed_query(prompt, model)
        search_filter = None
        if selected_course != 'Search in all documents':
            relevant = [selected_course]
            if text_input:
                relevant.append(f'{selected_course}_user_input')
            search_filter = {"file": {"$in": relevant}}
        results = search_pinecone(query_emb, pc, PINECONE_INDEX_NAME, namespace, top_k=3, filter=search_filter)
        context = "\n\n".join([r["metadata"]["text"] for r in results]) if results else ''
        answer = generate_answer_azure(
            prompt,
            context,
            selected_course,
            azure_endpoint=endpoint,
            azure_key=subscription_key,
            deployment_name=deployment)
    st.session_state["chat_history_all"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar="streamlit_utils/dasha.jpeg"):
        st.markdown(answer)

