import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import OPENROUTER_API_KEY

FAISS_DB_PATH = Path("./faiss_db")
EMBEDDING_MODEL = "openai/text-embedding-3-small"
_embeddings = None


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

    return _embeddings


def create_vectorstore(path: str):

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = [
        Document(
            page_content=f"Question: {item['question']}\nAnswer: {item['answer']}",
            metadata={
                "source": path,
                "question": item["question"],
                "answer": item["answer"],
            },
        )
        for item in data
    ]

    # For FAQ data, smaller chunks are better
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(
        docs,
        get_embeddings()
    )

    vectorstore.save_local(str(FAISS_DB_PATH))

    return vectorstore


def get_vectorstore():

    if not FAISS_DB_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run `python rag/ingest.py` before starting the app."
        )

    return FAISS.load_local(
        str(FAISS_DB_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=True
    )
