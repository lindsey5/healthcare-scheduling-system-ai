import json

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import OPENROUTER_API_KEY


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

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    vectorstore.save_local("./faiss_db")

    return vectorstore


def get_vectorstore():

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    return FAISS.load_local(
        "./faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )