import random
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
        answer = result.get("result", "")
        sources = result.get("source_documents", [])

        if not sources:
            return {
                "answer": {
                    "query": query,
                    "result": "I'm sorry, I couldn't find an exact answer, but I can try to help further if you rephrase your question."
                }
            }

        return {
            "answer": {
                "query": query,
                "result": answer
            },
            "sources": [doc.metadata for doc in sources]  # optional: show file/line
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
        categories = [
            "pregnancy prevention",
            "early signs of pregnancy",
            "safe sex practices",
            "family planning for teens",
            "myths about getting pregnant",
            "contraceptive awareness",
            "healthy pregnancy habits",
            "how to talk to a partner about safe sex",
            "understanding fertility cycles"
        ]
        selected_category = random.choice(categories)

        prompt = (
            f"Today is {formatted_date}. Give one practical, short, and medically accurate tip for adolescents "
            f"about {selected_category}. The tip should be helpful, clear, and specific — about 1–2 sentences. "
            f"Start directly with the tip. No introduction, no explanation."
        )
        
        result = qa_bot.invoke(prompt)
        tip = result["result"]

        # Cache and persist it
        tip_cache[today] = tip
        save_tip_cache()

    title = generate_title(tip)

    return {
        "date": formatted_date,
        "title": title["title"],
        "tip": title["tip"]
    }

@app.post("/insights")
def get_insights(data: InsightsRequest):
    try:
        # Create a unique hash from the symptoms and activities
        input_key = f"{data.sexDrives}-{data.moods}"
        input_hash = hashlib.md5(input_key.encode()).hexdigest()

        # Check cache
        if input_hash in insight_cache:
            cached = insight_cache[input_hash]
            return {
                "sexDrives": data.sexDrives,
                "moods": data.moods,
                "symptoms": data.symptoms,
                "vaginalDischarge": data.vaginalDischarge,
                "digestionAndStool": data.digestionAndStool,
                "pregnancyTest": data.pregnancyTest,
                "physicalActivity": data.physicalActivity,
                "insights": cached["insights"],
                "cached": True
            }

        # Format prompt
        sex_drives_str = ", ".join(data.sexDrives)
        moods_str = ", ".join(data.moods)
        symptoms_str = ", ".join(data.symptoms)
        vaginalDischarge_str = ", ".join(data.vaginalDischarge)
        digestionAndStool_str = ", ".join(data.digestionAndStool)
        pregnancyTest_str = ", ".join(data.pregnancyTest)
        physicalActivity_str = ", ".join(data.physicalActivity)

        prompt = (
            f"You are a helpful adolescent health assistant. "
            f"Based on the following sex drives: {sex_drives_str}, moods: {moods_str}, symptoms: {symptoms_str}, vaginalDischarge: {vaginalDischarge_str}, digestionAndStool: {digestionAndStool_str}, pregnancyTest: {pregnancyTest_str}, physicalActivity: {physicalActivity_str}"
            "generate specific, evidence-based insights that could indicate reproductive health outcomes "
            "such as early pregnancy signs, risks, or recommendations. "
            "Use only the information from official adolescent reproductive health documents such as "
            "'Sexual and Reproductive Health of Adolescents and Youth in the Philippines' and similar PDFs. "
            "Do not say you lack access to documents. Focus on accurate, specific, and medically sound advice relevant to teens."
        )

        result = qa_bot.invoke(prompt)
        insights_full = result["result"]

        # Summarize into detailed but structured output
        summary = summarize_insights(insights_full)

        # Cache it
        insight_cache[input_hash] = {
            "insights": {
                "full": insights_full,
                "summary": summary
            },
            "sexDrives": data.sexDrives,
            "moods": data.moods,
            "symptoms": data.symptoms,
            "vaginalDischarge": data.vaginalDischarge,
            "digestionAndStool": data.digestionAndStool,
            "pregnancyTest": data.pregnancyTest,
            "physicalActivity": data.physicalActivity
        }

        save_insight_cache()

        return {
            "sexDrives": data.sexDrives,
            "moods": data.moods,
            "symptoms": data.symptoms,
            "vaginalDischarge": data.vaginalDischarge,
            "digestionAndStool": data.digestionAndStool,
            "pregnancyTest": data.pregnancyTest,
            "physicalActivity": data.physicalActivity,
            "insights": {
                "full": insights_full,
                "summary": summary
            },
            "cached": False
        }

    except Exception as e:
        return {
            "error": str(e)
        }
    
def save_insight_cache():
    with open(INSIGHT_CACHE_FILE, "w") as f:
        json.dump(insight_cache, f)

def summarize_insights(insight_text: str) -> dict:
    # Basic parsing approach — extract categorized data
    summary = {
        "possibleConditions": [],
        "recommendations": [],
        "warnings": [],
        "notes": insight_text  # fallback full text
    }

    lines = insight_text.split("\n")
    for line in lines:
        lower = line.lower()
        if "may indicate" in lower or "suggests" in lower or "possible" in lower:
            summary["possibleConditions"].append(line.strip())
        elif "should" in lower or "recommended" in lower or "advised" in lower:
            summary["recommendations"].append(line.strip())
        elif "warning" in lower or "caution" in lower or "risk" in lower:
            summary["warnings"].append(line.strip())

    return summary
