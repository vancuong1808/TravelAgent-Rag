# 🧳 PBL6 Travel-Paper RAG

Hoi dap 21 paper travel planning + LLM agent (PBL6), co trich nguon paper + trang.

Chuyen tu notebook `Basic_Rag.ipynb` (1 PDF) -> project multi-doc that dung duoc.

## Cau truc

```
pbl6-travel-rag/
├── ingest.py      # nap 21 PDF -> chunk 800/150 -> ChromaDB (persist disk + metadata)
├── ask.py         # RAG core: retriever k=5 -> Gemini -> tra loi + trich nguon
├── app.py         # Streamlit demo
├── api.py         # FastAPI: POST /ask
├── eval/
│   ├── questions.json  # 20 cau eval
│   └── run_eval.py
├── data/          # (copy 1-2 PDF mau vao neu muon demo nhe)
├── requirements.txt
└── .env (.env.example mau)
```

## Chay (Windows)

```bat
cd C:\Users\ADMIN\Documents\code\pbl6-travel-rag
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
REM sua GOOGLE_API_KEY + PAPERS_DIR trong .env
python ingest.py
python ask.py "Paper nao dung multi-agent cho travel planning?"
streamlit run app.py
```

API:
```bat
uvicorn api:app --reload --port 8000
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"TREK danh gia agent bang metric gi?\"}"
```

## Eval

```bat
python eval/run_eval.py
```

Do thu cong: so cau co nguon / tong 20 cau. Nang cao: dung RAGAS
(`faithfulness`, `context_recall`) — xem `eval/run_eval.py`.

## Cai tien tu ban notebook cu

- Prompt cu: "khong co thi dung kien thuc cua ban" -> sua thanh chi tra loi tu context + tu choi khong bia.
- Retriever: k=2 -> k=5, chunk 1000/200 -> 800/150 (hop paper 2 cot), them metadata paper+page.
- Import deprecated `langchain_community HuggingFaceEmbeddings` -> `langchain_huggingface`.
- FAISS in-memory -> Chroma `persist_directory`, tat may khong mat index + loc duoc theo ten paper.
