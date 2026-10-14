import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


def main():
    # The script, .env, and PDF are all in the BUILDATHON folder.
    project_dir = Path(__file__).resolve().parent

    # Load environment configuration
    env_path = project_dir / ".env"

    if not env_path.is_file():
        raise FileNotFoundError(f".env file not found: {env_path}")

    load_dotenv(dotenv_path=env_path, override=True)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is missing from the .env file.")

    print("Environment configuration loaded.")

    # Define the user question
    question = "What is the main topic of the document?"
    print(f"\nQuestion: {question}")

    # Load the PDF document
    pdf_path = project_dir / "NetApp.pdf"

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    documents = PyPDFLoader(str(pdf_path)).load()

    if not documents:
        raise ValueError("The PDF contains no readable pages.")

    print(f"Loaded {len(documents)} pages.")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
    )
    splits = text_splitter.split_documents(documents)

    if not splits:
        raise ValueError("No text chunks were generated from the PDF.")

    print(f"Created {len(splits)} chunks.")

    # Generate Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    print("Embedding model initialized.")

    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("FAISS vector store created.")

    # Retrieve relevant document chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        raise ValueError("No relevant document chunks were retrieved.")

    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    print(f"Retrieved {len(relevant_docs)} relevant chunks.")

    # Initialize the OpenAI language model
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    print("OpenAI LLM initialized.")

    # Build the prompt and generate an answer
    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the retrieved context below.

If the context does not contain enough information, clearly state that
the answer cannot be determined from the retrieved content.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    # Display the result
    print("\n" + "=" * 55)
    print("RAG RESULT")
    print("=" * 55)
    print(f"\nQuestion:\n{question}")
    print(f"\nRetrieved Context:\n{context}")
    print(f"\nFinal Answer:\n{response.content}")
    print("\n" + "=" * 55)
    print("RAG PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 55)


if __name__ == "__main__":
    main()