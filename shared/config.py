#config.py
import os
from dotenv import load_dotenv
from Bio import Entrez
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase
import time

load_dotenv()

# ── NCBI ────────────────────────────────────────────────
Entrez.email   = os.getenv("NCBI_EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")

if not Entrez.email:
    raise ValueError("NCBI_EMAIL not found in .env file")

# ── Neo4j ────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j+s://your-instance.databases.neo4j.io")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def get_neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

# ── LLM ─────────────────────────────────────────────────
# llm_ner = ChatOllama(
#     model="gpt-oss:20b-cloud",
#     base_url="http://127.0.0.1:11434",  # ← your local Ollama server
#     headers={"Authorization": "Bearer " + os.getenv("OLLAMA_API_KEY", "")},
#     temperature=0
# )
# NER extraction

llm_ner= ChatOllama(model="llama3.1:8b",   temperature=0)  # NER extraction

llm_reasoning = ChatOllama(model="deepseek-r1:8b",   temperature=0)  # Reasoning/report


# ======================================================
# 🔥 Model Warmup
# Forces Ollama to fully load the model into VRAM before
# the pipeline starts firing parallel requests.
# Without this, parallel threads hit an unloaded model
# → CUDA error (status code: 500).
# ======================================================

def warmup_models(retries: int = 5, delay: float = 3.0):
    """
    Send a minimal dummy request to each model to ensure they are
    fully loaded into GPU memory before parallel inference begins.
    Retries on failure with a delay to allow Ollama time to recover.
    """
    models = [
        ("llm_ner (gpt-oss:20b-cloud)", llm_ner),
        ("llm_reasoning (deepseek-r1:8b)", llm_reasoning),
    ]

    for name, llm in models:
        print(f"🔥 Warming up {name}...", end=" ", flush=True)
        for attempt in range(1, retries + 1):
            try:
                llm.invoke("Say OK")
                print("✅ Ready")
                break
            except Exception as e:
                if attempt < retries:
                    print(f"\n   ⚠️ Attempt {attempt} failed ({e}). Retrying in {delay}s...", end=" ", flush=True)
                    time.sleep(delay)
                else:
                    print(f"\n   ❌ Warmup failed after {retries} attempts: {e}")
                    raise RuntimeError(
                        f"Model {name} failed to load. "
                        f"Check that Ollama is running and the model is pulled."
                    ) from e
