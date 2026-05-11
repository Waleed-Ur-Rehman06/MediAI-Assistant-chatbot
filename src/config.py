import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Centralized configuration for the MediAI Assistant."""
    
    # Flask settings
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-medical-ai")
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 8080))
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    # Model & Pinecone settings
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_ENV = os.environ.get("PINECONE_ENV", "us-east-1")
    INDEX_NAME = os.environ.get("INDEX_NAME", "medical-chatbot")
    MODEL_PATH = os.environ.get("MODEL_PATH", "model/llama-2-7b-chat.ggmlv3.q4_0.bin")
    
    # Rate Limiting
    RATE_LIMIT = "100 per day; 10 per minute"
    
    # LLM Settings
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 512))
    TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.1)) # Lower temperature for less hallucination
    
config = Config()
