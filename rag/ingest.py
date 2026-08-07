from rag.vectorstore import create_vectorstore
from config import *

if __name__ == "__main__":
    create_vectorstore("./data/datasets.json")
    print("Vector store created successfully.")