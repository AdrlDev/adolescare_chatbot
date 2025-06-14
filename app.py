from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from fastapi import FastAPI
from rag_bot import get_chatbot  # <-- your actual rag module
from rag_bot import generate_title
from rag_bot import save_tip_cache
from datetime import datetime
import hashlib

app = FastAPI()
qa_bot = get_chatbot()  # Re-enable this to use your RAG model

# Tip cache to store daily tips in memory
tip_cache = {}

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
        qa_bot = get_chatbot()
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