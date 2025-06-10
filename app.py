from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from fastapi import FastAPI
from rag_bot import get_chatbot  # <-- your actual rag module

app = FastAPI()
qa_bot = get_chatbot()  # Re-enable this to use your RAG model

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