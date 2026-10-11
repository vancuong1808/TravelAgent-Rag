"""Cau hinh dung chung: nap .env, gom path/model, validate som (fail fast).

Moi entrypoint (ingest/ask/api) import tu day de tranh hardcode va bao loi ro rang
khi thieu bien moi truong.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Path mac dinh portable: khong hardcode duong dan may ca nhan.
PAPERS_DIR = Path(os.getenv("PAPERS_DIR", "./data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "./chroma_db"))
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.8-flash")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def validate_env(require_papers: bool = False) -> None:
    """Bao loi ro rang + exit som khi thieu cau hinh bat buoc.

    - GOOGLE_API_KEY: moi entrypoint deu can (Gemini).
    - PAPERS_DIR: chi can khi ingest (require_papers=True).
    """
    problems = []
    if not os.getenv("GOOGLE_API_KEY"):
        problems.append(
            "GOOGLE_API_KEY chua duoc set. Copy .env.example -> .env va dien key."
        )
    if require_papers and not PAPERS_DIR.is_dir():
        problems.append(
            f"PAPERS_DIR khong ton tai: {PAPERS_DIR}. "
            "Tao folder va bo PDF vao, hoac sua PAPERS_DIR trong .env."
        )
    if problems:
        print("[config] Loi cau hinh:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        raise SystemExit(1)
