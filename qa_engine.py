# qa_engine.py

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

def load_vectorstore(path="/home/oluwafemi/Document-QA-BOT/.venv/Document-QA-BOT-RAG-LANGCHAIN/faiss_index"):
    """
    Load the FAISS index from disk.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def create_qa_chain(vectorstore, model_name="gpt-3.5-turbo"):
    """
    Create a RetrievalQA chain using the FAISS vectorstore and OpenAI chat model.
    """
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # or "map_reduce" or "refine"
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def ask_question(qa_chain, question):
    """
    Ask a question to the QA chain and return the response.
    """
    result = qa_chain(question)
    return result['result'], result['source_documents']
