from langchain.tools import tool

from rag.vectorstore import get_vectorstore
from rag.chain import create_rag_chain

vectorstore = get_vectorstore()
qa_chain = create_rag_chain(vectorstore)

@tool
def ask_question(question: str) -> str:
    """Search for knowledge-based answers related user's questions"""
    try:
        result = qa_chain.invoke({"query": question})
        print(result)
        return result["result"]
    except Exception as e:
            return f"{str(e)}"

def getChatbotTools():
    return [ask_question]