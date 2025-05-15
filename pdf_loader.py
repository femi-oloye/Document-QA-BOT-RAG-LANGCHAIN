from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(pdf_path, chunk_size = 1000, chunk_overlap = 200):
    # load the pdf
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # step 2: split the pdf into chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(pages)

    return chunks