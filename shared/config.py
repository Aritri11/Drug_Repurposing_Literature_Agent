import os
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
llm_ner       = ChatOllama(model="llama3.1:8b",   temperature=0)  # NER extraction
llm_reasoning = ChatOllama(model="deepseek-r1:8b",    temperature=0)  # Reasoning/report
