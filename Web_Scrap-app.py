
import os
import time
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import streamlit as st
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Tamil Nadu Agriculture Scheme Assistant",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Tamil Nadu Agriculture Scheme Assistant")
st.write(
    "Scrape scheme information from the official portal, "
    "store it locally, and ask questions using RAG."
)

PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
VECTORDB_DIR = PROJECT_DIR / "vectordb"

START_URL = (
    "https://www.tnagrisnet.tn.gov.in/people_app/GoScheme"
)

# Conservative crawl limits
MAX_PAGES = 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# --------------------------------------------------
# 2. Environment configuration
# --------------------------------------------------

def load_configuration():
    """Load the OpenAI key from the project's .env file."""

    if ENV_PATH.is_file():
        load_dotenv(ENV_PATH, override=False)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            f"OPENAI_API_KEY is missing. Check {ENV_PATH}"
        )


# --------------------------------------------------
# 3. Local embedding model
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# --------------------------------------------------
# 4. Playwright website scraper
# --------------------------------------------------

def is_allowed_url(url: str) -> bool:
    """Allow only HTML-page candidates within the target portal."""

    parsed = urlparse(url)

    return (
        parsed.scheme in ("http", "https")
        and parsed.netloc.lower() == "www.tnagrisnet.tn.gov.in"
        and parsed.path.startswith("/people_app/")
        and parsed.path.lower() != "/people_app/"
        and not parsed.path.lower().endswith(
            (".pdf", ".jpg", ".jpeg", ".png", ".gif",
             ".zip", ".doc", ".docx", ".xls", ".xlsx")
        )
    )


def scrape_website(start_url: str, max_pages: int = MAX_PAGES):
    """
    Scrape readable text from a limited number of same-portal
    pages using Playwright.

    Returns LangChain Documents with source metadata.
    """

    if not is_allowed_url(start_url):
        raise ValueError("The starting URL is outside the allowed portal.")

    documents = []
    visited = set()
    pending = [start_url]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)

        try:
            page = browser.new_page(
                locale="en-IN",
                ignore_https_errors=False,
            )

            page.set_default_navigation_timeout(30000)
            page.set_default_timeout(10000)

            while pending and len(visited) < max_pages:
                current_url = urldefrag(pending.pop(0))[0]

                if current_url in visited:
                    continue

                if not is_allowed_url(current_url):
                    continue

                visited.add(current_url)

                try:
                    response = page.goto(
                        current_url,
                        wait_until="domcontentloaded",
                    )

                    if response is None or response.status >= 400:
                        continue

                    # Allow client-side JavaScript to render content.
                    try:
                        page.locator("body").wait_for(
                            state="visible",
                            timeout=10000,
                        )
                    except PlaywrightTimeoutError:
                        continue

                    page.wait_for_timeout(1200)

                    title = page.title().strip() or current_url

                    # Extract readable text, including rendered tables.
                    text_content = page.locator("body").inner_text()
                    text_content = "\n".join(
                        line.strip()
                        for line in text_content.splitlines()
                        if line.strip()
                    )

                    if len(text_content) >= 50:
                        documents.append(
                            Document(
                                page_content=(
                                    f"Page title: {title}\n"
                                    f"Source URL: {current_url}\n\n"
                                    f"{text_content}"
                                ),
                                metadata={
                                    "source": current_url,
                                    "title": title,
                                },
                            )
                        )

                    # Discover links within the same portal.
                    links = page.locator("a[href]").evaluate_all(
                        """
                        elements => elements.map(a => ({
                            href: a.href,
                            text: (a.innerText || a.textContent || '').trim()
                        }))
                        """
                    )

                    for link in links:
                        href = urldefrag(
                            urljoin(current_url, link["href"])
                        )[0]

                        if (
                            is_allowed_url(href)
                            and href not in visited
                            and href not in pending
                        ):
                            pending.append(href)

                except PlaywrightTimeoutError:
                    st.warning(
                        f"Timed out while loading: {current_url}"
                    )

                except Exception as error:
                    st.warning(
                        f"Could not scrape {current_url}: {error}"
                    )

                # Be considerate to the government portal.
                time.sleep(0.5)

        finally:
            browser.close()

    if not documents:
        raise ValueError(
            "No readable page content was extracted. "
            "Check whether the website is reachable and whether "
            "its content is rendered or blocked."
        )

    return documents, len(visited)


# --------------------------------------------------
# 5. Split website content and create persistent FAISS
# --------------------------------------------------

def create_vectorstore(documents):
    """Chunk scraped pages, embed them, and save FAISS locally."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were created.")

    vectorstore = FAISS.from_documents(
        chunks,
        get_embeddings(),
    )

    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

    # Persist the index and associated document store.
    vectorstore.save_local(str(VECTORDB_DIR))

    return vectorstore, len(chunks)


def load_saved_vectorstore():
    """Load an existing local FAISS index, if available."""

    index_file = VECTORDB_DIR / "index.faiss"
    store_file = VECTORDB_DIR / "index.pkl"

    if not index_file.exists() or not store_file.exists():
        return None

    # Only load pickle files created by this trusted application.
    return FAISS.load_local(
        str(VECTORDB_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# --------------------------------------------------
# 6. RAG question answering
# --------------------------------------------------

def answer_question(vectorstore, question: str, top_k: int = 4):
    """Retrieve relevant passages and generate a grounded answer."""

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return "No relevant information was found.", []

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Title: {doc.metadata.get('title', 'Unknown')}\n"
        f"Content:\n{doc.page_content}"
        for doc in relevant_docs
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    prompt = f"""
You are an assistant for Tamil Nadu agricultural government schemes.

Answer the user's question using only the retrieved context.
Do not invent scheme names, subsidy amounts, eligibility criteria,
deadlines, or required documents.

If the context is insufficient, clearly state that the information
could not be verified from the indexed pages.

Distinguish official website content from any inference.
When relevant, identify the source page.

Retrieved context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, relevant_docs


# --------------------------------------------------
# 7. Session state and restore saved index
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = load_saved_vectorstore()
    except Exception as error:
        st.session_state.vectorstore = None
        st.warning(
            f"Could not load the saved index: {error}. "
            "You may need to scrape and re-index the website."
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_pages" not in st.session_state:
    st.session_state.indexed_pages = 0

if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0


# --------------------------------------------------
# 8. Sidebar: scrape and index
# --------------------------------------------------

with st.sidebar:
    st.header("Website ingestion")

    st.text_input(
        "Starting URL",
        value=START_URL,
        disabled=True,
    )

    st.caption(
        f"Maximum pages per run: {MAX_PAGES}"
    )

    if st.button(
        "Scrape and Index Website",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Scraping website with Playwright..."):
                documents, visited_count = scrape_website(START_URL)

            with st.spinner("Generating embeddings and saving FAISS..."):
                vectorstore, chunk_count = create_vectorstore(documents)

            st.session_state.vectorstore = vectorstore
            st.session_state.indexed_pages = len(documents)
            st.session_state.indexed_chunks = chunk_count
            st.session_state.messages = []

            st.success(
                f"Indexed {len(documents)} pages into "
                f"{chunk_count} chunks. "
                f"Visited {visited_count} URLs."
            )

        except Exception as error:
            st.error(f"Indexing failed: {error}")

    st.divider()

    if st.session_state.vectorstore is not None:
        st.success("Local FAISS index is available.")

        st.caption(f"Pages indexed: {st.session_state.indexed_pages}")
        st.caption(f"Chunks created: {st.session_state.indexed_chunks}")
        st.caption(f"Storage directory: {VECTORDB_DIR}")

    else:
        st.info(
            "No active index. Scrape the website to create one, "
            "or place a valid saved FAISS index in vectordb/."
        )

    st.divider()
    st.caption("Embeddings run locally; answers use the OpenAI API.")


# --------------------------------------------------
# 9. Main Streamlit chat
# --------------------------------------------------

if st.session_state.vectorstore is None:
    st.info(
        "Select **Scrape and Index Website** in the sidebar "
        "before asking questions."
    )

else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View retrieved source passages"):
                    for index, doc in enumerate(
                        message["sources"], start=1
                    ):
                        st.markdown(f"**Passage {index}**")
                        st.caption(
                            f"Title: {doc.metadata.get('title', 'Unknown')}"
                        )
                        st.markdown(
                            f"Source: {doc.metadata.get('source', 'Unknown')}"
                        )
                        st.write(doc.page_content)

    question = st.chat_input(
        "Ask about schemes, subsidies, eligibility, or documents..."
    )

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                load_configuration()

                with st.spinner("Searching indexed scheme information..."):
                    answer, source_docs = answer_question(
                        st.session_state.vectorstore,
                        question,
                    )

                st.markdown(answer)

                if source_docs:
                    with st.expander("View retrieved source passages"):
                        for index, doc in enumerate(
                            source_docs, start=1
                        ):
                            st.markdown(f"**Passage {index}**")
                            st.caption(
                                doc.metadata.get("title", "Unknown")
                            )
                            st.markdown(
                                f"Source: {doc.metadata.get('source', 'Unknown')}"
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
                error_message = f"Could not answer the question: {error}"
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
