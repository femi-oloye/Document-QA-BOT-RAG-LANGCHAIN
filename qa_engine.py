# qa_engine.py

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

import os
import streamlit as st

# Use Streamlit secrets if available, else fallback to env
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def load_vectorstore(path="/home/oluwafemi/Document-QA-BOT/.venv/Document-QA-BOT-RAG-LANGCHAIN/faiss_index"):
    """
    Load the FAISS index from disk.
    """
    load_dotenv()
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
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
    output_key="answer"  # 👈 THIS tells memory to store only 'answer'
)


    qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=False,
    output_key="answer"  # ✅ FIXED: Comma added before this line
)

    return qa_chain, memory

def ask_question_with_memory(chain, question):
    """
    Ask a question to the chain and get context-aware answer.
    """
    result = chain.invoke({"question": question})
    return result['answer'], result['source_documents']
