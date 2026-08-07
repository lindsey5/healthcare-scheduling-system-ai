from dotenv import load_dotenv

from llm.model import get_healthcare_ai_model

# Load environment variables from .env if running locally
load_dotenv()
llm = get_healthcare_ai_model()

print(llm.invoke("Hello"))