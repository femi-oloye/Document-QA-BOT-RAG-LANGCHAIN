# vector_store.py

import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def embed_and_store(chunks, save_path="faiss_index"):
    """
    Embed the document chunks using OpenAI and store them in a FAISS index.

    Parameters:
        chunks (List[Document]): The text chunks to embed.
        save_path (str): Directory to save the FAISS index.

    Returns:
        FAISS: A FAISS vector store instance.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is missing. Please check your .env file.")

    # Step 1: Create embeddings (API key is auto-read from env)
    embeddings = OpenAIEmbeddings()

    # Step 2: Embed and store in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Step 3: Save FAISS index locally
    vectorstore.save_local(save_path)

    return vectorstore
