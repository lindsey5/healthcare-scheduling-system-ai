import json

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def create_vectorstore(path: str) -> Chroma:
    # Load JSON
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Convert JSON to LangChain Documents
    documents = [
        Document(
            page_content=f"Question: {item['question']}\nAnswer: {item['answer']}",
            metadata={
                "question": item["question"],
            },
        )
        for item in data
    ]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
    )

    print(documents)
    docs = splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )

    return vectorstore


def get_vectorstore() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )