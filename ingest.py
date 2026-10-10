"""Ingest 21 paper PBL6 -> chunk -> ChromaDB (persist disk + metadata)."""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
PAPERS_DIR = Path(os.getenv("PAPERS_DIR", r"C:/Users/ADMIN/Downloads/pbl6_paper"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "./chroma_db"))

def main():
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    print(f"Tim thay {len(pdfs)} PDF trong {PAPERS_DIR}")
    assert pdfs, "Khong tim thay PDF nao. Kiem tra PAPERS_DIR trong .env"
    all_docs = []
    for pdf in pdfs:
        docs = PyPDFLoader(str(pdf)).load()
        for d in docs:
            d.metadata["paper"] = pdf.stem[:80]
            d.metadata["source_file"] = pdf.name
        all_docs.extend(docs)
    print(f"Tong {len(all_docs)} pages.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    print(f"Chia thanh {len(chunks)} chunks.")
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Chroma tu persist xuong disk — khong can save_local nhu FAISS
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=str(INDEX_DIR),
    )
    print(f"Luu Chroma tai {INDEX_DIR} ({vs._collection.count()} vectors).")

if __name__ == "__main__":
    main()
