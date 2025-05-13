import streamlit as st
import faiss
import numpy as np
import pickle
import json
import requests
from sentence_transformers import SentenceTransformer

# ✅ OpenRouter API Key
API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ✅ Load Resources
@st.cache_resource
def load_resources():
    index = faiss.read_index("/content/drive/MyDrive/Books for Text extraction/All medicines HOMEO A2Z encyclopedia/chunks and embeddings of above ALL medicine/faiss_index.idx")

    with open("/content/drive/MyDrive/Books for Text extraction/All medicines HOMEO A2Z encyclopedia/chunks and embeddings of above ALL medicine/metadata.pkl", "rb") as f:
        docs = pickle.load(f)

    with open("/content/drive/MyDrive/Books for Text extraction/All medicines HOMEO A2Z encyclopedia/chunks and embeddings of above ALL medicine/id_to_meta.json", "r") as f:
        id_to_meta = json.load(f)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return index, docs, id_to_meta, embedder

# Load once
index, docs, id_to_meta, embedder = load_resources()

# ✅ Function to call OpenRouter API
def call_openrouter_api(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://yourwebsite.com",  # Optional
        "X-Title": "Homeopathy AI Assistant"  # Optional
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload))
    response_data = response.json()

    if response.status_code == 200:
        return response_data["choices"][0]["message"]["content"].strip()
    else:
        return f"❌ Error from OpenRouter: {response_data.get('error', 'Unknown error')}"

# ✅ Streamlit UI
st.title("🔍 Homeopathy AI Assistant")
query = st.text_input("Enter your case or query:", placeholder="e.g. Homeopathic remedies for suppressed emotions")

if st.button("Get Remedies") and query:
    with st.spinner("Thinking..."):
        # Step 1: Embed query
        query_embedding = embedder.encode([query])

        # Step 2: Search FAISS
        D, I = index.search(np.array(query_embedding), k=5)

        # Step 3: Build Context
        context = ""
        for idx in I[0]:
            meta = id_to_meta[str(idx)]
            chunk = docs[idx]
            context += f"Source: {meta['source']}, Chunk ID: {meta['chunk_id']}\n{chunk}\n---\n"

        # Step 4: Construct Prompt for OpenRouter
        prompt = f"""
You are a highly experienced classical homeopathic physician with deep understanding of Hahnemann’s Organon of Medicine, Kent’s Repertory, Allen’s Keynotes, Boericke’s Materia Medica, and other classical texts.

You will now receive a detailed patient case containing symptoms from physical, mental, emotional, and general domains.

Your task is to:
1. Identify key characteristic symptoms and convert them into appropriate rubrics using classical repertories.
2. Briefly list important rubrics that define the uniqueness of the case.
3. Based on the repertorized symptoms and modalities, narrow down a few top remedy options (DO NOT mention remedy names yet).
4. Show how a professional homeopath would cross-check these options in Allen’s Keynotes and Materia Medica.
5. Use reasoning to select the most suitable remedy based on mental, physical, general, and modality alignment.
6. Suggest suitable potency and repetition advice based on the chronicity, depth, and vitality of the case.

now mention Best remedies in list.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Query: {query}

Answer (with sources):
"""

        # Step 5: Get Response from OpenRouter
        response_text = call_openrouter_api(prompt)

        st.markdown("### 🧠 AI Response")
        st.markdown(response_text)
