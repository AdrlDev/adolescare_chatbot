from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from fastapi import FastAPI
from rag_bot import get_chatbot  # <-- your actual rag module
from rag_bot import generate_title
from rag_bot import save_tip_cache
from datetime import datetime
from rag_bot import InsightsRequest
from rag_bot import INSIGHT_CACHE_FILE
import hashlib

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load insight cache on startup
    if INSIGHT_CACHE_FILE.exists():
        with open(INSIGHT_CACHE_FILE) as f:
            insight_cache.update(json.load(f))
    yield  # Let the app run
    # You could add cleanup code here on shutdown

app = FastAPI(lifespan=lifespan)
qa_bot = get_chatbot()  # Re-enable this to use your RAG model

# Tip cache to store daily tips in memory
tip_cache = {}
insight_cache = {}

import json
from pathlib import Path

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Adolescare RAG Chatbot is live."}

@app.get("/chat")
def chat(query: str = Query(...)):
    try:
        result = qa_bot.invoke(query)
        answer = result['result']
        sources = result.get('source_documents', [])

        if not sources:
            return {
                "answer": {
                    "query": query,
                    "result": "Sorry, I couldn't find information about that in the documents provided."
                }
            }

        return {
            "answer": {
                "query": query,
                "result": answer
            }
        }

    except Exception as e:
        return {
            "answer": {
                "query": query,
                "result": f"Error: {str(e)}"
            }
        }
    
@app.get("/todays-tip")
def get_todays_tip():
    today = datetime.today().strftime("%Y-%m-%d")
    formatted_date = datetime.today().strftime("%B %d, %Y")

    # Check tip cache
    if today in tip_cache:
        tip = tip_cache[today]
    else:
        # Generate new tip using LangChain QA
        prompt = "Give me one practical, short, and helpful health or self-care tip for adolescents. Be specific and clear."
        result = qa_bot.invoke(prompt)
        tip = result["result"]

        # Cache and persist it
        tip_cache[today] = tip
        save_tip_cache()

    title = generate_title(tip)

    return {
        "date": formatted_date,
        "title": title,
        "tip": tip
    }

@app.post("/insights")
def get_insights(data: InsightsRequest):
    try:
        # Create a unique hash from the symptoms and activities
        input_key = f"{data.symptoms}-{data.activities}"
        input_hash = hashlib.md5(input_key.encode()).hexdigest()

        # Check cache
        if input_hash in insight_cache:
            cached = insight_cache[input_hash]
            return {
                "symptoms": data.symptoms,
                "activities": data.activities,
                "insights": cached["insights"],
                "cached": True
            }

        # Format prompt
        symptoms_str = ", ".join(data.symptoms)
        activities_str = ", ".join(data.activities)

        prompt = (
            f"You are a helpful adolescent health assistant. "
            f"Based on the following symptoms: {symptoms_str}, and activities: {activities_str}, "
            "generate specific, evidence-based insights that could indicate reproductive health outcomes "
            "such as early pregnancy signs, risks, or recommendations. "
            "Use only the information from official adolescent reproductive health documents such as "
            "'Sexual and Reproductive Health of Adolescents and Youth in the Philippines' and similar PDFs. "
            "Do not say you lack access to documents. Focus on accurate, specific, and medically sound advice relevant to teens."
        )

        result = qa_bot.invoke(prompt)
        insights = result["result"]

        # Cache it
        insight_cache[input_hash] = {
            "insights": insights,
            "symptoms": data.symptoms,
            "activities": data.activities
        }

        save_insight_cache()

        return {
            "symptoms": data.symptoms,
            "activities": data.activities,
            "insights": insights,
            "cached": False
        }

    except Exception as e:
        return {
            "error": str(e)
        }
    
def save_insight_cache():
    with open(INSIGHT_CACHE_FILE, "w") as f:
        json.dump(insight_cache, f)