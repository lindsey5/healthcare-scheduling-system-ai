from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from rag.vectorstore import get_vectorstore
from llm.model import get_openrouter_model

def create_rag_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3 })

    qa_chain = RetrievalQA.from_chain_type(
        llm=get_openrouter_model(),  
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain

_vectorstore = None
_qa_chain = None


def get_qa_chain():
    global _vectorstore, _qa_chain

    if _qa_chain is None:
        _vectorstore = get_vectorstore()
        _qa_chain = create_rag_chain(_vectorstore)

    return _qa_chain