import os
from dotenv import load_dotenv
from Bio import Entrez
from langchain_ollama import ChatOllama
from pydantic import BaseModel, RootModel
from typing import List

# ======================================================
# 🔐 Load Environment Variables
# ======================================================
load_dotenv()

Entrez.email = os.getenv("NCBI_EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")

if not Entrez.email:
    raise ValueError("NCBI_EMAIL not found in .env file")


# ======================================================
# 🧠 LLM Setup
# ======================================================
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

# ======================================================
# 📦 Structured Output Schema (Pydantic v2)
# ======================================================

from enum import Enum

class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"

class EvidenceType(str, Enum):
    BASELINE = "BASELINE"
    INTERVENTION = "INTERVENTION"

class GeneDirection(BaseModel):
    gene: str
    direction: Direction
    evidence_type: EvidenceType

class GeneDirectionList(RootModel[List[GeneDirection]]):
    pass

structured_llm = llm.with_structured_output(GeneDirectionList)


# ======================================================
# 📚 Step 1: Fetch PubMed Abstracts
# ======================================================

def fetch_pubmed_abstracts(disease: str, max_results: int = 5):
    search_term = f"""
    {disease}[Title/Abstract]
    AND
    ("expression level" OR upregulated OR downregulated)
    """

    search_handle = Entrez.esearch(
        db="pubmed",
        term=search_term,
        retmax=max_results
    )

    search_results = Entrez.read(search_handle)
    search_handle.close()

    id_list = search_results["IdList"]

    if not id_list:
        print("No PubMed results found.")
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


# ======================================================
# 🧬 Step 2: Extract Baseline Dysregulated Genes
# ======================================================

def extract_baseline_dysregulation(disease: str, abstracts: str):
    if not abstracts:
        return []

    prompt = f"""
    From the following PubMed abstracts about {disease},

    Identify genes whose expression levels are altered in patients with {disease}
    under baseline disease conditions (NOT after drug treatment).

    STRICT RULES:
    - Extract only genes dysregulated in disease patients compared to controls.
    - Ignore genes altered due to drug treatment or experimental manipulation.
    - If gene is upregulated/activated/positively regulated (turning on gene expression) in disease → direction = "UP"
    - If gene is downregulated/suppressed/negatively regulated (turning off gene expression) in disease → direction = "DOWN"
    - evidence_type must be:
        "BASELINE" if naturally dysregulated in disease
        "INTERVENTION" if altered due to treatment or compound

    Return structured output only.
    """

    try:
        result = structured_llm.invoke(prompt + "\n\n" + abstracts)
        return [item.model_dump(mode="json") for item in result.root]
    except Exception as e:
        print("Structured extraction failed:", e)
        return []


# ======================================================
# 🚿 Step 3: Filter Only Baseline Evidence
# ======================================================

def filter_baseline_only(extracted_data: list):
    return [
        item for item in extracted_data
        if item["evidence_type"] == "BASELINE"
    ]


# ======================================================
# 🧪 Test Run
# ======================================================

if __name__ == "__main__":

    disease = "Parkinson's disease"

    print("\n🔎 Fetching PubMed abstracts...")
    abstracts = fetch_pubmed_abstracts(disease, max_results=5)

    print("\n🧠 Extracting gene dysregulation...")
    extracted = extract_baseline_dysregulation(disease, abstracts)

    print("\n📦 Raw Extracted:")
    print(extracted)

    baseline_only = filter_baseline_only(extracted)

    print("\n✅ Baseline Dysregulated Genes Only:")
    print(baseline_only)