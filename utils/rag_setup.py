import streamlit as st
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource(show_spinner="Setting up Knowledge Base... (This might take a minute on first run)")
def setup_rag_pipeline(pdf_path="resnet_paper.pdf"):
    if not os.path.exists(pdf_path):
        return None, None, f"PDF file not found at {pdf_path}. Please ensure it was downloaded properly."

    # Create persistent storage paths
    persist_dir = "./chroma_db"
    bm25_path = "./bm25_retriever.pkl"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check if we have both cached stores
    if os.path.exists(persist_dir) and os.path.exists(bm25_path):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
    else:
        # 1. Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
            
        # 2. Split into conceptual chunks (smaller to avoid model context length issues)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        splits = text_splitter.split_documents(pages)
        
        # 3. Create Vector Search and BM25 Search
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_dir)
        
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 2
        
        # Save BM25 retriever
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
            
    # Load dense retriever from vectorstore
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # RRF (Reciprocal Rank Fusion) function
    def retrieve_with_rrf(query, k=60):
        dense_docs = dense_retriever.invoke(query)
        sparse_docs = bm25_retriever.invoke(query)
        
        # Combine using RRF
        rrf_scores = {}
        for rank, doc in enumerate(dense_docs):
            if doc.page_content not in rrf_scores:
                rrf_scores[doc.page_content] = {"doc": doc, "score": 0.0}
            rrf_scores[doc.page_content]["score"] += 1.0 / (rank + k)
            
        for rank, doc in enumerate(sparse_docs):
            if doc.page_content not in rrf_scores:
                rrf_scores[doc.page_content] = {"doc": doc, "score": 0.0}
            rrf_scores[doc.page_content]["score"] += 1.0 / (rank + k)
            
        # Sort by score and take top 2
        ranked_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked_docs[:2]]
    
    # 4. Simple local QA setup (using explicit model and tokenizer to avoid pipeline task inference issues)
    # Using a larger Flan-T5 base model to improve QA inference quality and prevent degradation across queries.
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    # Create a simple wrapper to mimic the pipeline's behavior
    def llm_pipeline(prompt_text):
        inputs = tokenizer(prompt_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=200)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": decoded}]
    
    return retrieve_with_rrf, llm_pipeline, None

def generate_answer(query, retriever, llm_pipeline):
    # Retrieve relevant context using our custom RRF
    docs = retriever(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Simple Synthesis Prompt
    prompt = f"Answer the question based only on the context below. If you don't know, just say you don't know.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Generate Output safely
    try:
        response = llm_pipeline(prompt)
        answer_text = response[0]['generated_text']
    except Exception as e:
        answer_text = f"An error occurred during generation: {e}"
        
    return answer_text, docs
