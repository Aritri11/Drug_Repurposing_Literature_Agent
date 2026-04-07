#config.py
import os
import time
from dotenv import load_dotenv
from Bio import Entrez
from langchain_ollama import ChatOllama
from neo4j import GraphDatabase

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
# keep_alive=-1 tells Ollama to NEVER unload this model from VRAM.
# Without this, Ollama resets the timer on every inference request
# back to the default 5 minutes — causing mid-pipeline unloads.
llm_ner       = ChatOllama(model="llama3.1:8b",    temperature=0, keep_alive=-1)
llm_reasoning = ChatOllama(model="deepseek-r1:8b", temperature=0, keep_alive=-1)


# ======================================================
# 🔥 Model Warmup
# ======================================================

def free_reasoning_model():
    """
    Unload deepseek-r1:8b from VRAM before the NER phase starts.
    During NER, only llama3.1:8b is needed. Keeping deepseek-r1:8b
    loaded consumes ~5GB VRAM and leaves no room for llama3.1:8b
    to reload if it gets evicted, causing 60s+ reload timeouts.
    Call this after warmup, before graph.invoke().
    """
    import requests
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "deepseek-r1:8b", "keep_alive": 0, "prompt": "", "stream": False},
            timeout=10
        )
        if r.status_code == 200:
            print("🧹 deepseek-r1:8b unloaded from VRAM — freeing space for NER phase")
        else:
            print(f"⚠️  Could not unload deepseek-r1:8b: {r.status_code}")
    except Exception as e:
        print(f"⚠️  deepseek-r1:8b unload failed (non-fatal): {e}")


def warmup_models(retries: int = 5, delay: float = 3.0):
    """
    Send a minimal dummy request to each model to ensure they are
    fully loaded into GPU memory before parallel inference begins.
    The keep_alive=-1 in the ChatOllama definition means every
    warmup call also sets the model to never-unload.
    """
    models = [
        ("llm_ner (llama3.1:8b)",          llm_ner),
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
