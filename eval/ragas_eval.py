"""Eval RAG bang RAGAS that: faithfulness, answer_relevancy, context_precision, context_recall.

Chay:  .venv/Scripts/python.exe eval/ragas_eval.py [--n 5] [--k 5]
Ket qua: eval/results.csv + eval/results.md (bang diem dua vao README/CV).

Luu y tuong thich: langchain-community>=0.4 bo module `chat_models.vertexai`
nhung ragas 0.4.3 van import no -> doan shim duoi day map sang package
`langchain-google-vertexai` (chi dung luc import, khong anh huong diem so).
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

# --- Shim tuong thich ragas 0.4.3 + langchain-community 0.4 (xem docstring) ---
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ImportError:
    try:
        import langchain_google_vertexai as _v
        _m = types.ModuleType("langchain_community.chat_models.vertexai")
        _m.ChatVertexAI = _v.ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = _m
    except ImportError:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ask import ask  # noqa: E402

from datasets import Dataset  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from ragas import RunConfig, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

def build_dataset(n: int, k: int) -> Dataset:
    qs = json.loads(Path(__file__).with_name("questions.json").read_text(encoding="utf-8"))
    qs = qs[:n] if n else qs
    rows = []
    for i, item in enumerate(qs, 1):
        print(f"[{i}/{len(qs)}] hoi: {item['q'][:60]}...")
        ans, srcs = ask(item["q"], k=k)
        rows.append({
            "question": item["q"],
            # ground_truth hien la keyword-hint (du de chay context_precision/recall).
            # Muon chuan hon: viet dap an mau day du cho tung cau.
            "ground_truth": item.get("hint", ""),
            "answer": ans,
            "contexts": [d.page_content for d in srcs],
        })
    return Dataset.from_list(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="So cau chay thu (0 = tat ca 20)")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    ds = build_dataset(args.n, args.k)
    judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=os.getenv("MODEL_NAME", "gemini-3.8-flash"), temperature=0))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    print("Cham bang RAGAS...")
    # Retry khi Gemini 503 qua tai (gap chieu nay khi test)
    run_cfg = RunConfig(max_retries=10, max_wait=60.0)
    result = evaluate(ds, metrics=METRICS, llm=judge, embeddings=emb, run_config=run_cfg)
    df = result.to_pandas()
    out_csv = Path(__file__).with_name("results.csv")
    out_md = Path(__file__).with_name("results.md")
    df.to_csv(out_csv, index=False)
    means = df.select_dtypes(include="number").mean(numeric_only=True)
    lines = ["# RAGAS eval (TravelAgent-RAG)", "",
             f"Chay {len(df)} cau, retriever k={args.k}.", "",
             "| metric | mean |", "|---|---|"]
    for col in means.index:
        lines.append(f"| {col} | {means[col]:.3f} |")
    lines += ["",
              "Co cot: sample_index | question | ground_truth(hint) | answer | contexts | ...scores.",
              "Chi tiet tung cau xem `results.csv`."]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nLuu: {out_csv} , {out_md}")

if __name__ == "__main__":
    main()
