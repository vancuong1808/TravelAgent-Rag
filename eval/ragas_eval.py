"""Chấm điểm RAG bằng RAGAS: faithfulness, answer_relevancy, context_precision, context_recall.
Chạy:  .venv/Scripts/python.exe eval/ragas_eval.py [--n 5] [--k 5]
Kết quả: eval/results.csv + eval/results.md (bảng điểm dùng cho README/CV).
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Ẩn vài cảnh báo deprecation đã biết (không ảnh hưởng kết quả).
warnings.filterwarnings("ignore", message="Importing .* from 'ragas.metrics' is deprecated")
warnings.filterwarnings("ignore", message="LangchainLLMWrapper is deprecated")
warnings.filterwarnings("ignore", message="LangchainEmbeddingsWrapper is deprecated")

# Console Windows mặc định cp1252 -> không in được tiếng Việt; ép stdout UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ask import ask

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def load_questions(n: int = 0) -> list[dict]:
    """Đọc câu hỏi từ eval/questions.json; n>0 thì chỉ lấy n câu đầu."""
    path = Path(__file__).with_name("questions.json")
    questions = json.loads(path.read_text(encoding="utf-8"))
    return questions[:n] if n else questions


def build_dataset(n: int, k: int) -> Dataset:
    """Gọi ask() cho từng câu hỏi rồi đóng gói thành Dataset chuẩn RAGAS."""
    questions = load_questions(n)
    rows = []
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] hỏi: {item['q'][:60]}...")
        answer, sources = ask(item["q"], k=k)
        rows.append({
            "question": item["q"],
            # ground_truth hiện là keyword-hint (đủ để chạy context_precision/recall).
            # Muốn chuẩn hơn: viết đáp án mẫu đầy đủ cho từng câu trong questions.json.
            "ground_truth": item.get("hint", ""),
            "answer": answer,
            "contexts": [d.page_content for d in sources],
        })
    return Dataset.from_list(rows)


def make_judge() -> LangchainLLMWrapper:
    """LLM giám khảo: Qwen local (LM Studio), không dùng Gemini để khỏi hết quota."""
    from langchain_openai import ChatOpenAI
    base = ChatOpenAI(
        model=os.getenv("LM_STUDIO_MODEL", "qwen3-8b"),
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        api_key="lm-studio",
        temperature=0,
    )
    return LangchainLLMWrapper(base)


def make_embeddings() -> LangchainEmbeddingsWrapper:
    """Embeddings dùng cho answer_relevancy."""
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL))


def write_reports(df, k: int) -> tuple[Path, Path]:
    """Ghi results.csv và results.md (mean từng metric); trả về đường dẫn 2 file."""
    out_csv = Path(__file__).with_name("results.csv")
    out_md = Path(__file__).with_name("results.md")
    df.to_csv(out_csv, index=False)

    means = df.select_dtypes(include="number").mean(numeric_only=True)
    lines = [
        "# RAGAS eval (TravelAgent-RAG)", "",
        f"Chạy {len(df)} câu, retriever k={k}.", "",
        "| metric | mean |", "|---|---|",
    ]
    for col in means.index:
        lines.append(f"| {col} | {means[col]:.3f} |")
    lines += [
        "",
        "Cột: question | ground_truth(hint) | answer | contexts | ...scores.",
        "Chi tiết từng câu xem `results.csv`.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return out_csv, out_md


def main() -> None:
    """Entry point: build dataset -> chấm RAGAS -> xuất CSV/MD."""
    ap = argparse.ArgumentParser(description="Chấm RAGAS cho TravelAgent-RAG")
    ap.add_argument("--n", type=int, default=0, help="Số câu chạy thử (0 = tất cả)")
    ap.add_argument("--k", type=int, default=5, help="Số context retriever lấy về")
    args = ap.parse_args()

    ds = build_dataset(args.n, args.k)
    print(f"Chấm {len(ds)} câu bằng RAGAS (4 metrics)...")

    # Retry khi Gemini quá tải (503) -- ragas tự thử lại theo RunConfig.
    run_cfg = RunConfig(max_workers=1, timeout=300, max_wait=300, max_retries=2)
    result = evaluate(
        ds,
        metrics=METRICS,
        llm=make_judge(),
        embeddings=make_embeddings(),
        run_config=run_cfg,
        raise_exceptions=False,
    )

    out_csv, out_md = write_reports(result.to_pandas(), args.k)
    print(f"\nLưu: {out_csv} , {out_md}")


if __name__ == "__main__":
    main()
