from langchain_ollama import ChatOllama

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