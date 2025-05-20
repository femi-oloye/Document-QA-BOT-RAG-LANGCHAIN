# qa_engine.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env if running locally
load_dotenv()

# 🔐 Securely get the OpenAI API key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# ❗ Raise error if key is missing
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Set it in Streamlit secrets or .env.")

# Set the environment variable for OpenAI use
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def load_vectorstore(path="faiss_index"):
    """
    Load the FAISS vectorstore index from disk.
    """
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def create_qa_chain_with_memory(vectorstore, model_name="gpt-3.5-turbo"):
    """
    Create a ConversationalRetrievalChain with memory using FAISS and OpenAI Chat Model.
    """
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    retriever = vectorstore.as_retriever()
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # ✅ Tell memory to store only the final answer
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        output_key="answer"  # ✅ Fix output key error
    )

    return qa_chain, memory


def ask_question_with_memory(chain, question):
    """
    Ask a question to the chain and get context-aware answer and source documents.
    """
    result = chain.invoke({"question": question})
    return result["answer"], result["source_documents"]
