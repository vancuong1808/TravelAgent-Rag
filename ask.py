"""RAG core: load ChromaDB -> retriever k=5 (+ filter paper) -> Gemini + trich nguon.

LLM, embeddings, vectorstore va chain duoc cache (singleton/lru_cache) de tai
su dung giua cac request thay vi rebuild moi lan.
"""
from functools import lru_cache
from operator import itemgetter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import EMBED_MODEL, INDEX_DIR, MODEL_NAME, validate_env

validate_env()  # fail fast neu thieu GOOGLE_API_KEY (api import ask cung huong loi nay)

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


@lru_cache(maxsize=1)
def get_llm():
    """Singleton LLM: Qwen local lam chinh, Gemini fallback (MODEL_BACKEND)."""
    import os
    gemini = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
    if os.getenv("MODEL_BACKEND", "qwen").lower() == "gemini":
        return gemini
    from langchain_openai import ChatOpenAI
    qwen = ChatOpenAI(
        model=os.getenv("LM_STUDIO_MODEL", "qwen3-8b"),
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
        api_key="lm-studio",  # LM Studio khong can key, nhung langchain yeu cau
        temperature=0,
    )
    return qwen.with_fallbacks([gemini])


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Singleton vectorstore: nap embeddings + Chroma 1 lan duy nhat."""
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=str(INDEX_DIR), embedding_function=emb)


@lru_cache(maxsize=32)
def build_chain(k: int = 5, paper: str | None = None):
    """Chain + retriever cho cap (k, paper); cache theo tham so."""
    search_kwargs: dict = {"k": k}
    if paper:
        search_kwargs["filter"] = {"paper": paper}
    retriever = get_vectorstore().as_retriever(search_kwargs=search_kwargs)
    chain = (
        {"context": itemgetter("input") | retriever | docs_to_str,
         "input": itemgetter("input")}
        | PROMPT | get_llm() | StrOutputParser()
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
