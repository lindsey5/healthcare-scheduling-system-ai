from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from config import OPENROUTER_API_KEY

def get_qwen3_4b_model():
    return ChatOllama(
        model="qwen3:4b",
        temperature=0,
        keep_alive="30m",
    )

def get_healthcare_ai_model():
    return ChatOllama(
        model="healthcare-ai", 
        temperature=0,
        keep_alive="30m", 
    )

def get_openrouter_model():
    return ChatOpenAI(
        model="openrouter/free",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )
