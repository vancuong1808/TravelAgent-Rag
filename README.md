# 🧳 PBL6 Travel-Paper RAG

Hoi dap 21 paper travel planning + LLM agent (PBL6), co trich nguon paper + trang.

Chuyen tu notebook `Basic_Rag.ipynb` (1 PDF) -> project multi-doc that dung duoc.

## Kien truc

```
                         data/*.pdf  (21 paper PBL6)
                              │
                              │  PyPDFLoader
                              ▼
                       [ ingest.py ]
              chunk 800/150  +  metadata(paper, page)
                              │
                              ▼
                     chroma_db/  (Chroma persist)
                              │    embedding: all-MiniLM-L6-v2
                              ▼
   question ──>  [ ask.py ]  retriever k=5 (+ filter theo paper)
                              │
                              ├──> Gemini (MODEL_NAME) ──> answer + sources
                              │
                              ├──> [ app.py ]  Streamlit demo
                              └──> [ api.py ]  FastAPI  POST /ask
```

## Cau truc

```
pbl6-travel-rag/
├── config.py      # nap .env + path/model dung chung + validate som
├── ingest.py      # nap 21 PDF -> chunk 800/150 -> ChromaDB (persist disk + metadata)
├── ask.py         # RAG core: retriever k=5 -> Gemini -> tra loi + trich nguon (cache LLM/retriever)
├── app.py         # Streamlit demo
├── api.py         # FastAPI: POST /ask
├── eval/
│   ├── questions.json  # 20 cau eval
│   ├── run_eval.py     # kiem tra nhanh co nguon/khong (khong can RAGAS)
│   └── ragas_eval.py   # cham 4 metric RAGAS
├── data/          # bo PDF vao day (khong commit PDF; giu .gitkeep)
├── requirements.txt
└── .env (.env.example mau)
```

## Chay (Windows)

```bat
git clone <repo-url> && cd pbl6-travel-rag
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
REM sua GOOGLE_API_KEY trong .env, va PAPERS_DIR tro toi folder chua PDF (mac dinh ./data)
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

Chay nhanh: `python eval/run_eval.py` (kiem tra co nguon/khong).
Chuan RAGAS: `python eval/ragas_eval.py` (20 cau, 4 metric
`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`)
-> ra `eval/results.csv + results.md`, copy bang mean vao day.

## Ket qua eval (baseline, 5 cau, Qwen-3-8B + Gemini judge)

| metric | mean |
|---|---|
| faithfulness | 0.558 |
| answer_relevancy | 0.372 |
| context_precision | 0.107 |
| context_recall | 0.467 |

Nhan xet: baseline thap, ro nhat o `context_precision` — retriever
dang tra ve chunk nhieu. Huong cai thien: viet `ground_truth` day du
thay keyword-hint, tang k / thu chunk nho hon, them reranker, chay du 20 cau
bang backend Gemini roi lay so cuoi. (Khong dung so baseline nay tren CV.)
Thu nhanh 2 cau: `python eval/ragas_eval.py --n 2`.
Neu gap 503 UNAVAILABLE la do Google qua tai, doi chut chay lai.

## Ket qua RAGAS

| metric | mean |
|---|---|
| faithfulness | _chay `python eval/ragas_eval.py` de dien_ |
| answer_relevancy | |
| context_precision | |
| context_recall | |

Bang so that duoc sinh tu dong vao `eval/results.md` sau khi chay eval.

## Cai tien tu ban notebook cu

- Prompt cu: "khong co thi dung kien thuc cua ban" -> sua thanh chi tra loi tu context + tu choi khong bia.
- Retriever: k=2 -> k=5, chunk 1000/200 -> 800/150 (hop paper 2 cot), them metadata paper+page.
- Import deprecated `langchain_community HuggingFaceEmbeddings` -> `langchain_huggingface`.
- FAISS in-memory -> Chroma `persist_directory`, tat may khong mat index + loc duoc theo ten paper.
- Cache LLM/embeddings/retriever (`lru_cache`) thay vi rebuild moi request; path portable, validate env som.
