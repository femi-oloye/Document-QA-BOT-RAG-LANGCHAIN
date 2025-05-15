# Document QA Bot with lANGCHAIN + OPENAI 
## steps used to create this project

✅ Step 1: Load and Chunk the PDF
🔸 What i am Doing:

I will:

    Load a PDF file

    Split the document into smaller, meaningful chunks

    Prepare it for embedding in the next step

✅ Step 3: Embed Chunks with OpenAI + FAISS
🔸 What i am Doing:

i will:

    Use OpenAI’s embedding model to convert each chunk into a numeric vector

    Store these vectors in a FAISS vector store so we can perform similarity search later