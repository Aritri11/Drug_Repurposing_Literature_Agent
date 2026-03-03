import os
import json
from dotenv import load_dotenv
from Bio import Entrez
from langchain_ollama import ChatOllama
from graph_backend import Neo4jGraph
from test_file import graph
from pydantic import BaseModel, RootModel
from typing import List

# --------------------------------------------------
# 🔐 Load environment variables
# --------------------------------------------------
load_dotenv()

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL")

if not NCBI_EMAIL:
    raise ValueError("NCBI_EMAIL not found in .env file")

Entrez.email = NCBI_EMAIL

if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY
else:
    print("⚠️ Warning: No NCBI API key found. Using limited rate mode.")

# --------------------------------------------------
# 🧠 LLM Setup
# --------------------------------------------------
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

class GeneDirection(BaseModel):
    gene: str
    direction: str  # "UP" or "DOWN"

class GeneDirectionList(RootModel[List[GeneDirection]]):
    pass

structured_llm = llm.with_structured_output(GeneDirectionList)

# --------------------------------------------------
# 📚 Step 1: Fetch PubMed Abstracts
# --------------------------------------------------
def fetch_pubmed_abstracts(disease: str, max_results: int = 5):
    try:
        search_handle = Entrez.esearch(
            db="pubmed",
            term=f"{disease} AND gene expression",
            retmax=max_results
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results["IdList"]

        if not id_list:
            return ""

        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=id_list,
            rettype="abstract",
            retmode="text"
        )

        abstracts = fetch_handle.read()
        fetch_handle.close()

        return abstracts

    except Exception as e:
        print("Error fetching PubMed:", e)
        return ""


# --------------------------------------------------
# 🧬 Step 2: Extract Gene + Direction via LLM
# --------------------------------------------------
def extract_gene_directions(disease: str, abstracts: str):
    if not abstracts:
        return []

    prompt = f"""
    From the following abstracts about {disease},
    extract gene names and whether they are UP or DOWN regulated.

    Direction must be exactly either "UP" or "DOWN".
    """

    try:
        result = structured_llm.invoke(prompt + "\n\n" + abstracts)
        return [item.dict() for item in result.__root__]
    except Exception as e:
        print("Structured extraction failed:", e)
        return []


# --------------------------------------------------
# 🧱 Step 3: Update Neo4j Graph
# --------------------------------------------------
def update_disease_gene_graph(disease: str, extracted_data: list):
    for item in extracted_data:
        gene = item["gene"]
        direction = item["direction"]

        query = """
        MERGE (d:Disease {name:$disease})
        MERGE (g:Gene {name:$gene})
        MERGE (d)-[r:ALTERS_EXPRESSION]->(g)
        SET r.direction = $direction,
            r.evidence_count = COALESCE(r.evidence_count, 0) + 1,
            r.confidence_score =
                CASE
                    WHEN COALESCE(r.evidence_count, 0) >= 10 THEN 1.0
                    ELSE (COALESCE(r.evidence_count, 0) / 10.0)
                END
        """

        with graph.driver.session() as session:
            session.run(
                query,
                disease=disease,
                gene=gene,
                direction=direction
            )


# --------------------------------------------------
# 🚀 Main Ingestion Function
# --------------------------------------------------
def ingest_disease_expression(disease: str):
    print(f"\n🔎 Fetching PubMed data for: {disease}")

    abstracts = fetch_pubmed_abstracts(disease)
    extracted = extract_gene_directions(disease, abstracts)

    if extracted:
        update_disease_gene_graph(disease, extracted)
        print("✅ Graph updated successfully.")
    else:
        print("⚠️ No gene-direction pairs extracted.")

    return extracted

result = ingest_disease_expression("Alzheimer's disease")

print("\nExtracted Data:")
print(result)