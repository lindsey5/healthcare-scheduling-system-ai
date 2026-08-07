from langchain.chat_models import init_chat_model
from config import *

def get_gemini_model():
    return init_chat_model("gemini-2.5-flash", model_provider="google_genai")