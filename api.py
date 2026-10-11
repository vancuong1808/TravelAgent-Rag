from fastapi import FastAPI
from pydantic import BaseModel
from ask import ask

app = FastAPI(title="PBL6 Travel RAG API")

class AskIn(BaseModel):
    question: str
    k: int = 5

@app.post("/ask")
def ask_ep(body: AskIn):
    ans, srcs = ask(body.question, k=body.k)
    return {
        "answer": ans,
        "sources": [
            {"paper": s.metadata.get("paper"), "file": s.metadata.get("source_file"),
             "page": s.metadata.get("page")} for s in srcs
        ],
    }

@app.get("/health")
def health():
    return {"ok": True}
