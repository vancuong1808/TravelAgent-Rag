"""Streamlit demo: hoi dap 21 paper PBL6."""
import streamlit as st
from ask import ask

st.set_page_config(page_title="PBL6 Travel RAG", page_icon="🧳")
st.title("🧳 PBL6 Travel-Paper RAG")
st.caption("Hoi dap 21 paper travel planning + LLM agent, co trich nguon.")

q = st.text_input("Cau hoi:", placeholder="VD: TREK danh gia agent bang metric gi?")
k = st.slider("So chunk truy hoi (k)", 2, 10, 5)
if st.button("Hoi", type="primary") and q.strip():
    with st.spinner("Dang truy hoi..."):
        ans, srcs = ask(q, k=k)
    st.markdown("### Tra loi")
    st.write(ans)
    st.markdown("### Nguon")
    for d in srcs:
        st.caption(f"📄 {d.metadata.get('source_file')} — p.{d.metadata.get('page')}")
