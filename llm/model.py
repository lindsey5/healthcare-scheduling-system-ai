from langchain.chat_models import init_chat_model
from config import *
from langchain_ollama import ChatOllama

def get_gemini_model():
    return init_chat_model("gemini-3.6-flash", model_provider="google_genai")

def get_ollama_model():
    return ChatOllama(
        model="qwen3:4b",
        temperature=0,
    )