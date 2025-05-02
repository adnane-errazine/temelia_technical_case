import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


print("Environment variables loaded successfully.")
