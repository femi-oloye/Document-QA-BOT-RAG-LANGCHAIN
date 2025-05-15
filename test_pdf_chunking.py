# test_qa.py

from qa_engine import load_vectorstore, create_qa_chain, ask_question

vectorstore = load_vectorstore()
qa_chain = create_qa_chain(vectorstore)

question = "What is this PDF about?"  # Change to fit your PDF
answer, sources = ask_question(qa_chain, question)

print("Answer:", answer)
print("\nSources:")
for doc in sources:
    print("-", doc.metadata.get("source", "Unknown"))
