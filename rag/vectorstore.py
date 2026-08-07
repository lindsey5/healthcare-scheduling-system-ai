import json

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def create_vectorstore(path: str):

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = [
        Document(
            page_content=f"Question: {item['question']}\nAnswer: {item['answer']}",
            metadata={
                "question": item["question"]
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

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    vectorstore.save_local("./faiss_db")

    return vectorstore


def get_vectorstore():

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    return FAISS.load_local(
        "./faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )