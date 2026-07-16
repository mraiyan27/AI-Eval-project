import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Embedding
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# -------------------------
# Gemini (Generator)
# -------------------------

GENERATION_MODEL = "gemini-2.5-flash"

# -------------------------
# Groq (Judge)
# -------------------------

JUDGE_MODEL = "llama-3.3-70b-versatile"

# Chroma
CHROMA_DB_PATH = "./chroma_db"

# Retrieval
DEFAULT_K = 5
BROAD_QUERY_K = 15
FETCH_K = 30

# Files
CHAT_HISTORY = "evaluations/chat_history.csv"
SCORED_RESULTS = "evaluations/scored_results.csv"
QUESTIONS = "evaluations/questions.csv"

# Phoenix
USE_PHOENIX = False