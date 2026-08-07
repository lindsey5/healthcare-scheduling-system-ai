import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env if running locally
load_dotenv()
llm = GoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.environ["GEMINI_API_KEY"],
)

print(llm.invoke("Hello"))