from fastapi import FastAPI
import os
from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

from pydantic import BaseModel
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

import re

app = FastAPI()

INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_index_metadata.pkl"
VECTORSTORE_PATH = "vectorstore.index"

import json
from pathlib import Path

TIP_CACHE_FILE = "tips.json"
INSIGHT_CACHE_FILE = Path("insight_cache.json")

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

def load_documents():
    pdf_files = [
        "documents/dswd_teenage_pregnancy_guidelines.pdf",
        "documents/module_on_reproductive_health.pdf",
        "documents/operational_guidance_2014_eng_clinical_standards_manual_family_planning.pdf",
        "documents/POPCOMXII_AHDModule_Preventing_Teenage_Pregnancy.pdf",
        "documents/sexual_and_reproductive_health_and_rights_of_young_people_in_asia_and_the_pacific.pdf",
        "documents/sexual_and_reproductive_health_of_adolescents_and_youth_in_the_philippines.pdf",
        "documents/teenage_pregnancy.pdf",
        "documents/teenpreg.pdf",
    ]

    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        raw_docs = loader.load()

        # ✅ Add metadata to each page/document
        for doc in raw_docs:
            doc.metadata["source"] = pdf

        chunks = splitter.split_documents(raw_docs)
        documents.extend(chunks)

    return documents

def get_vectorstore():
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=api_key)

    if os.path.exists(VECTORSTORE_PATH):
        print("Loading cached vectorstore...")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new vectorstore and saving cache...")
        documents = load_documents()
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)

    return vectorstore

def get_chatbot():
    vectorstore = get_vectorstore()
    chat = ChatCohere(model="command-a-03-2025", temperature=0, cohere_api_key=api_key)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5}
    )
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        retriever=retriever,
        return_source_documents=True  # return sources to verify relevance
    )
    return qa

def generate_title(tip: str) -> str:
    try:
        # Clean the tip (remove markdown like "**Tip:**" or "Tip:")
        tip = re.sub(r'^\s*\*?\*?Tip:\*?\*?\s*', '', tip, flags=re.IGNORECASE)
        tip = re.sub(r'\*+', '', tip).strip()
        tip = tip[0].upper() + tip[1:] if tip else tip

        chat = ChatCohere(
            model="command-a-03-2025",
            temperature=0.2,
            cohere_api_key=api_key
        )
        prompt = f"Give a short, 3- to 5-word title for this adolescent health tip: \"{tip}\""
        result = chat.invoke(prompt)

        if hasattr(result, "content"):
            result = result.content

        # Remove filler phrases like "Sure! Here's a title:"
        cleaned = re.sub(
            r'(?i)^.*?(?:title\s*:|is\s*|here\s*(?:is|\'s)?\s*(?:a)?\s*title\s*[:\-]?)\s*',
            '',
            result
        )

        # Remove quotes and emoji/symbols
        cleaned = re.sub(r'[^\w\s]', '', cleaned)               # remove punctuation/symbols
        cleaned = re.sub(r'[\U00010000-\U0010ffff]', '', cleaned)  # remove emojis

        # Capitalize each word and limit to 5 words max
        words = cleaned.strip().split()
        title_words = words[:5]
        title = ' '.join(word.capitalize() for word in title_words)

        return {"title": title, "tip": tip}

    except Exception as e:
        print(f"[ERROR] Failed to generate title with LLM: {e}")
        return {
            "title": "Daily Health Tip",
            "tip": tip.strip()
        }

# Load tips from file at startup
def load_tip_cache():
    global tip_cache
    if Path(TIP_CACHE_FILE).exists():
        with open(TIP_CACHE_FILE, "r") as file:
            try:
                tip_cache = json.load(file)
            except json.JSONDecodeError:
                tip_cache = {}
    else:
        tip_cache = {}

# Save updated tips to file
def save_tip_cache():
    with open(TIP_CACHE_FILE, "w") as file:
        json.dump(tip_cache, file, indent=2)

# Call this once at startup
load_tip_cache()

class InsightsRequest(BaseModel):
    symptoms: List[str]
    activities: List[str]