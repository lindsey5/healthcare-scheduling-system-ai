import os
from dotenv import load_dotenv

# Load environment variables from .env if running locally
load_dotenv()

API_URL = os.getenv("API_URL")