"""
German Tax Declaration RAG Chatbot (Streamlit)
================================================
A single‑file Streamlit application that:
  • Scrapes the content of https://handbookgermany.de/en/tax-declaration
  • Builds (or loads) a FAISS vector store
  • Uses LangChain retrieval‑augmented generation with either OpenAI or a local Llama model

Run:
    pip install -r requirements.txt
    streamlit run tax_declaration_chatbot.py

Environment variables (choose one model family):
    # OpenAI
    export OPENAI_API_KEY="sk‑..."

    # OR Llama‑cpp
    export LLAMA_CPP_MODEL="/path/to/llama‑2‑7b‑chat.gguf"

Optional: Set HTTP proxy vars if you need network access when scraping.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain

################################################################################
# Configuration
################################################################################

TARGET_URL = "https://handbookgermany.de/en/tax-declaration"
DATA_DIR = Path("data")
VECTOR_PATH = DATA_DIR / "faiss_index"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

################################################################################
# Utility functions
################################################################################

def scrape_page(url: str) -> str:
    """Return cleaned visible text from the given URL."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    return "\n\n".join(paragraphs)


def get_embeddings():
    """Return an embedding model (OpenAI or HuggingFace fallback)."""
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings()
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore(text: str, embeddings) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c, metadata={"source": TARGET_URL}) for c in chunks]
    vectorstore = FAISS.from_documents(docs, embeddings)
    DATA_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTOR_PATH))
    return vectorstore


@st.cache_resource(show_spinner="Initialising vector store – first run may take up to a minute…")
def load_or_create_vectorstore():
    embeddings = get_embeddings()
    if VECTOR_PATH.exists():
        return FAISS.load_local(
            str(VECTOR_PATH),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    text = scrape_page(TARGET_URL)
    return build_vectorstore(text, embeddings)


def get_llm():
    """Return a chat LLM (OpenAI or local llama‑cpp)."""
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(temperature=0)
    model_path = os.getenv("LLAMA_CPP_MODEL", "models/llama-2-7b-chat.gguf")
    return LlamaCpp(model_path=model_path, n_ctx=4096, temperature=0, verbose=False)

################################################################################
# Streamlit UI
################################################################################

def main():
    st.set_page_config(page_title="German Tax Declaration Assistant", page_icon="💶")
    st.title("💶 German Tax Declaration Assistant")

    st.markdown(
        """
        Ask anything about filing a **Steuererklärung** in Germany.  
        Answers come **only** from the content of the
        [Handbook Germany tax‑declaration page](https://handbookgermany.de/en/tax-declaration).
        """,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[tuple[str, str]] = []

    user_question = st.text_input("Your question", placeholder="e.g. Who needs to file a tax declaration?", key="q")

    if st.button("Ask", use_container_width=True) and user_question:
        vectordb = load_or_create_vectorstore()
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        llm = get_llm()

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            return_source_documents=True,
        )

        result = qa_chain({"question": user_question, "chat_history": st.session_state.chat_history})
        answer: str = result["answer"]
        sources = {doc.metadata["source"] for doc in result["source_documents"]}

        st.markdown("**Answer**")
        st.write(answer)

        with st.expander("Sources"):
            for s in sources:
                st.write(s)

        st.session_state.chat_history.append((user_question, answer))


if __name__ == "__main__":
    main()
