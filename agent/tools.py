import httpx
from langchain.tools import tool
from rag.chain import get_qa_chain
from config import API_URL

@tool
def ask_question(question: str) -> str:
    """Search for knowledge-based answers related Bagumbayan Health Center"""
    try:
        result = get_qa_chain().invoke({"query": question})
        print(result)
        return result["result"]
    except Exception as e:
            return f"{str(e)}"

@tool
def get_services() -> dict:
    """Get the healthcare services offered by the health center, sort it by day."""

    response = httpx.get(
        f"{API_URL}/api/services",
        timeout=10,
    )

    response.raise_for_status()

    return response.json()

def getChatbotTools():
    return [ask_question, get_services]