import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

from llm.model import get_qwen3_4b_model

# Load environment variables from .env if running locally
load_dotenv()
llm = get_qwen3_4b_model()

print(llm.invoke("Hello"))