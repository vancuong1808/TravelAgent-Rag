"""RAG core: load ChromaDB -> retriever k=5 (+ filter paper) -> Gemini + trich nguon."""
import os
from pathlib import Path
from operator import itemgetter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
INDEX_DIR = Path(os.getenv("INDEX_DIR", "./chroma_db"))

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Ban la tro ly nghien cuu PBL6 (travel planning + LLM agents).\n"
     "Chi tra loi DUA TREN ngu canh duoc cap. Moi y phai ghi ro nguon [ten paper - trang X].\n"
     "Neu ngu canh khong du, noi thang: 'Tai lieu PBL6 chua de cap' va khong bia.\n\n"
     "Ngu canh:\n{context}"),
    ("human", "{input}"),
])

def docs_to_str(docs):
    out = []
    for d in docs:
        paper = d.metadata.get("paper", "?")
        page = d.metadata.get("page", "?")
        out.append(f"[{paper} - p.{page}]\n{d.page_content}")
    return "\n\n".join(out)

def build_chain(k: int = 5, paper: str | None = None):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma(
        persist_directory=str(INDEX_DIR),
        embedding_function=emb,
    )
    search_kwargs: dict = {"k": k}
    if paper:
        # Loc metadata: chi truy hoi trong 1 paper (demo suc manh Chroma vs FAISS)
        search_kwargs["filter"] = {"paper": paper}
    retriever = vs.as_retriever(search_kwargs=search_kwargs)
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
    chain = (
        {"context": itemgetter("input") | retriever | docs_to_str,
         "input": itemgetter("input")}
        | PROMPT | llm | StrOutputParser()
    )
    return chain, retriever

def ask(question: str, k: int = 5, paper: str | None = None):
    chain, retriever = build_chain(k, paper)
    srcs = retriever.invoke(question)
    ans = chain.invoke({"input": question})
    return ans, srcs

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Paper nao dung multi-agent cho travel planning?"
    a, s = ask(q)
    print(a)
    print("\n--- NGUON ---")
    for d in s:
        print("-", d.metadata.get("source_file"), "p.", d.metadata.get("page"))
