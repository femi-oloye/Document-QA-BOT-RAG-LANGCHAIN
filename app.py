# app.py

import streamlit as st
from pdf_loader import load_and_chunk_pdf
from vector_store import embed_and_store
from qa_engine import load_vectorstore, create_qa_chain_with_memory, ask_question_with_memory
import os

st.set_page_config(page_title="Document QA Bot", layout="wide")

st.title("Document QA Chatbot with Conversational RAG ")
st.markdown("Ask follow-up questions with context using Retrieval-Augmented Generation + memory.")

# --- Sidebar ---
with st.sidebar:
    st.header("🔑 API Configuration")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# --- Main Logic ---
if uploaded_file and openai_api_key:
    with st.spinner("🔍 Processing your document..."):
        # Save uploaded file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Step 1: Load and chunk
        chunks = load_and_chunk_pdf("temp.pdf")

        # Step 2: Embed and store in FAISS
        vectorstore = embed_and_store(chunks)

        # Step 3: Load vectorstore and create conversational QA chain
        vectorstore = load_vectorstore()
        qa_chain, memory = create_qa_chain_with_memory(vectorstore)

    st.success("✅ Document is ready! Start chatting below.")

    # --- Chat UI ---
    st.subheader("💬 Ask a Question")
    question = st.text_input("Type your question here:")

    if question:
        with st.spinner("🔎 Thinking..."):
            answer, sources = ask_question_with_memory(qa_chain, question)
            st.markdown(f"**Answer:** {answer}")

            with st.expander("📚 Sources"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:** {doc.page_content[:300]}...")

            with st.expander("🧠 Chat Memory"):
                for msg in memory.chat_memory.messages:
                    st.markdown(f"**{msg.type.capitalize()}**: {msg.content}")
else:
    st.info("👈 Please enter your OpenAI API key and upload a PDF to begin.")
