from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from llm.model import get_healthcare_ai_model

def create_rag_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3 })

    qa_chain = RetrievalQA.from_chain_type(
        llm=get_healthcare_ai_model(),  
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain