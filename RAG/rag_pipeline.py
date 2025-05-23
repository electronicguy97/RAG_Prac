from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 문서 로드 및 전처리
def load_documents(json_path):
    loader = JSONLoader(file_path=json_path, jq_schema='.[] | {page_content: .text, metadata: {case_id: .case_id, type: .type, timestamp: .timestemp}}', text_content=False)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    return splitter.split_documents(docs)

# 벡터 스토어 생성
def create_vectorstore(docs):
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    db = Chroma.from_documents(docs, embedding=embedding_model, persist_directory="./chroma_db")
    db.persist()
    return db

# 모델 로드
def load_local_llm():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=True)
    return HuggingFacePipeline(pipeline=pipe)
# RAG QA Chain 생성
def get_qa_chain(db):
    llm = load_local_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_type="similarity", k=3)
    )
