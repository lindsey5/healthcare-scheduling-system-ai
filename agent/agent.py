from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent  

from agent.tools import getChatbotTools
from llm.model import get_ollama_model

_chat_bot_agent = None
_model = None

def initialize_agent():
    """
    Initialize model and agents.
    """
    global _chat_bot_agent, _model

    _memory = InMemorySaver()
    _model = get_ollama_model()

    chat_bot_prompt = """
You are an AI assistant for the Barangay Bagumbayan Health Center.

Rules:
- Answer only questions related to the Barangay Bagumbayan Health Center.
- Be polite and professional.
- If you don't know the answer, say you don't have that information.
- Keep answers concise and accurate.
- Do not guess—always rely on the tools to answer
"""
    if _chat_bot_agent is None:
        _chat_bot_agent = create_react_agent(
            model=_model,
            tools=getChatbotTools(),
            prompt=chat_bot_prompt,
            checkpointer=_memory
        )

def get_chat_bot_agent():
    """Getter for chat bot agent"""
    return _chat_bot_agent