import streamlit as st
from rag_pipeline import load_documents, create_vectorstore, get_qa_chain
import os

st.set_page_config(page_title="RAG Test", layout="wide")

st.title("🔍 한국어 문서 기반 RAG QA 시스템")

if 'qa_chain' not in st.session_state:
    with st.spinner("문서 로딩 중..."):
        docs = load_documents("docs.json")
        db = create_vectorstore(docs)
        st.session_state.qa_chain = get_qa_chain(db)

query = st.text_input("질문을 입력하세요", placeholder=" 개인정보 수집 시 법적 조건은?")
if st.button("질문하기") and query:
    with st.spinner("생성 중..."):
        result = st.session_state.qa_chain.run(query)
        st.markdown("### 답변")
        st.write(result)
