import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Streamlit page setup
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="centered",
)

st.title("📄 PDF Question Answering")
st.write("Upload a PDF, index it, and ask questions about its contents.")


# --------------------------------------------------
# Environment configuration
# --------------------------------------------------

def load_configuration():
    """Load the API key from .env in the same folder as this script."""
    project_dir = Path(__file__).resolve().parent
    env_path = project_dir / ".env"

    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            f"OPENAI_API_KEY was not found. Add it to this file: {env_path}"
        )


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# --------------------------------------------------
# PDF processing and vector store
# --------------------------------------------------

def create_vectorstore(pdf_bytes: bytes, filename: str):
    """Load a PDF, split it into chunks, and create a FAISS index."""
    suffix = Path(filename).suffix or ".pdf"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        documents = PyPDFLoader(temp_path).load()

        if not documents:
            raise ValueError("The PDF contains no readable pages.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks could be extracted from the PDF.")

        vectorstore = FAISS.from_documents(chunks, get_embeddings())
        return vectorstore, len(documents), len(chunks)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# RAG question answering
# --------------------------------------------------

def answer_question(vectorstore, question: str, number_of_chunks: int = 3):
    """Retrieve relevant text and ask the model to answer from that text."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": number_of_chunks},
    )
    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return (
            "I couldn't find relevant information in the document.",
            [],
        )

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in relevant_docs
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    prompt = f"""You are a document question-answering assistant.

Answer the question using only the retrieved context below.
If the context does not contain enough information, clearly state that
the answer cannot be determined from the retrieved content.
Do not make up facts.

Retrieved context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content, relevant_docs


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# --------------------------------------------------
# Sidebar: upload and index a PDF
# --------------------------------------------------

with st.sidebar:
    st.header("Your document")

    uploaded_pdf = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
    )

    if st.button(
        "Index PDF",
        type="primary",
        disabled=uploaded_pdf is None,
        use_container_width=True,
    ):
        try:
            load_configuration()

            with st.spinner("Reading and indexing the PDF..."):
                vectorstore, page_count, chunk_count = create_vectorstore(
                    uploaded_pdf.getvalue(),
                    uploaded_pdf.name,
                )

            st.session_state.vectorstore = vectorstore
            st.session_state.document_name = uploaded_pdf.name
            st.session_state.messages = []

            st.success(
                f"Indexed {page_count} pages into {chunk_count} text chunks."
            )

        except Exception as error:
            st.error(f"Could not index the PDF: {error}")

    if st.session_state.document_name:
        st.caption(f"Indexed: **{st.session_state.document_name}**")

    st.divider()
    st.caption("Your OpenAI API key is read from `.env` beside this script.")


# --------------------------------------------------
# Main chat area
# --------------------------------------------------

if st.session_state.vectorstore is None:
    st.info("Upload a PDF in the sidebar and select **Index PDF** to begin.")
else:
    # Show earlier messages after Streamlit reruns.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View retrieved passages"):
                    for index, doc in enumerate(message["sources"], start=1):
                        page_number = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Passage {index} · Page {page_number}**")
                        st.write(doc.page_content)

    question = st.chat_input("Ask a question about your PDF")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                load_configuration()

                with st.spinner("Searching the PDF..."):
                    answer, source_docs = answer_question(
                        st.session_state.vectorstore,
                        question,
                    )

                st.markdown(answer)

                if source_docs:
                    with st.expander("View retrieved passages"):
                        for index, doc in enumerate(source_docs, start=1):
                            page_number = doc.metadata.get("page", 0) + 1
                            st.markdown(
                                f"**Passage {index} · Page {page_number}**"
                            )
                            st.write(doc.page_content)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_docs,
                    }
                )

            except Exception as error:
                error_message = f"Sorry, I couldn't answer that: {error}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )