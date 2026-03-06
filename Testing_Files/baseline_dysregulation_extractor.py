import os
from dotenv import load_dotenv
from Bio import Entrez
from langchain_ollama import ChatOllama
from pydantic import BaseModel, RootModel
from typing import List
from tqdm import tqdm
import mygene




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
    pmid: str

class GeneDirectionList(RootModel[List[GeneDirection]]):
    pass

structured_llm = llm.with_structured_output(GeneDirectionList)


def fetch_pubmed_abstracts(disease: str, max_results: int = 20):
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
        return []

    # Fetch as XML to cleanly separate abstracts
    fetch_handle = Entrez.efetch(
        db="pubmed",
        id=id_list,
        rettype="xml",
        retmode="xml"
    )

    records = Entrez.read(fetch_handle)
    fetch_handle.close()

    abstract_list = []

    # Safely parse the XML for abstracts
    for article in records['PubmedArticle']:
        try:
            pmid = str(article['MedlineCitation']['PMID'])  # ← extract PMID
            abstract_text = article['MedlineCitation']['Article']['Abstract']['AbstractText']
            # Sometimes abstracts are split into sections (Background, Methods, etc.)
            full_abstract = " ".join([str(text) for text in abstract_text])
            abstract_list.append({"pmid": pmid, "abstract": full_abstract})
        except KeyError:
            # Skip if the paper has no abstract
            continue

    return abstract_list


# ======================================================
# 🧬 Step 2: Extract Baseline Dysregulated Genes (One by One)
# ======================================================

def extract_baseline_dysregulation(disease: str, abstract_list: list):
    all_extracted = []

    if not abstract_list:
        return []

    prompt = f"""
    From the following PubMed abstract about {disease},

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
    - For pmid field, use exactly this value: {{pmid_placeholder}}

    Return structured output only.
    """

    # Wrap the list in tqdm for a nice progress bar
    for i, item in enumerate(tqdm(abstract_list, desc="Processing Abstracts")):
        pmid = item["pmid"]  # ← unpack
        abstract = item["abstract"]  # ← unpack
        try:
            filled_prompt = prompt.replace("{pmid_placeholder}", pmid)  # ← inject pmid
            # We process ONE abstract at a time
            result = structured_llm.invoke(filled_prompt + "\n\nAbstract:\n" + abstract)

            # Check if it actually found anything
            if result and result.root:
                for entry in result.root:
                    data = entry.model_dump(mode="json")
                    data["pmid"] = pmid  # ← force correct pmid (don't trust LLM)
                    all_extracted.append(data)

        except Exception as e:
            print(f"\nStructured extraction failed on abstract {i + 1}: {e}")

    return all_extracted


# ======================================================
# 🚿 Step 3: Filter Only Baseline Evidence
# ======================================================

def filter_baseline_only(extracted_data: list):
    return [
        item for item in extracted_data
        if item["evidence_type"] == "BASELINE"
    ]

mg = mygene.MyGeneInfo()

def validate_genes_against_mygene(extracted_data):
    validated = []

    for item in extracted_data:
        gene = item["gene"]

        try:
            result = mg.query(gene, species="human", size=1)

            if result["hits"]:
                official_symbol = result["hits"][0].get("symbol")

                if official_symbol:
                    item["gene"] = official_symbol  # Normalize symbol
                    validated.append(item)

        except Exception as e:
            print(f"Gene validation failed for {gene}: {e}")

    return validated

# ======================================================
# 🧪 Test Run
# ======================================================

# if __name__ == "__main__":
#
#     disease = "Parkinson's disease"
#
#     print("\n🔎 Fetching PubMed abstracts...")
#     abstracts = fetch_pubmed_abstracts(disease, max_results=20)
#
#     print("\n🧠 Extracting gene dysregulation...")
#     extracted = extract_baseline_dysregulation(disease, abstracts)
#
#     print("\n📦 Raw Extracted:")
#     print(extracted)
#
#     baseline_only = filter_baseline_only(extracted)
#     validated_genes = validate_genes_against_mygene(baseline_only)
#
#     # print("\n✅ Validated Genes (Official Symbols Only):")
#     # print(validated_genes)
#
#     print("\n✅ Baseline Dysregulated Genes Only:")
#     print(validated_genes)

def run_extraction_pipeline(disease: str, max_results: int = 20):
    abstracts = fetch_pubmed_abstracts(disease, max_results)
    extracted = extract_baseline_dysregulation(disease, abstracts)
    baseline_only = filter_baseline_only(extracted)
    validated_genes = validate_genes_against_mygene(baseline_only)
    return validated_genes


if __name__ == "__main__":
    disease = "Parkinson's disease"
    results = run_extraction_pipeline(disease)
    print(results)

####################################################################################################################################
#
# def fetch_pubmed_records(disease: str, max_results: int = 5):
#     search_term = f"""
#     {disease}[Title/Abstract]
#     AND
#     ("expression level" OR upregulated OR downregulated)
#     """
#
#     search_handle = Entrez.esearch(
#         db="pubmed",
#         term=search_term,
#         retmax=max_results
#     )
#
#     search_results = Entrez.read(search_handle)
#     search_handle.close()
#
#     id_list = search_results["IdList"]
#
#     if not id_list:
#         print("No PubMed results found.")
#         return []
#
#     fetch_handle = Entrez.efetch(
#         db="pubmed",
#         id=id_list,
#         rettype="medline",
#         retmode="text"
#     )
#
#     records_text = fetch_handle.read()
#     fetch_handle.close()
#
#     # Split individual records
#     records = records_text.strip().split("\n\nPMID- ")
#
#     structured_records = []
#
#     for record in records:
#         if not record.strip():
#             continue
#
#         lines = record.split("\n")
#
#         title = ""
#         abstract = ""
#         pmid = ""
#
#         for line in lines:
#             if line.startswith("TI  -"):
#                 title = line.replace("TI  -", "").strip()
#             elif line.startswith("AB  -"):
#                 abstract = line.replace("AB  -", "").strip()
#             elif line.startswith("PMID-"):
#                 pmid = line.replace("PMID-", "").strip()
#
#         structured_records.append({
#             "title": title,
#             "abstract": abstract,
#             "pmid": pmid
#         })
#
#     return structured_records
#
#
# def extract_baseline_dysregulation(disease: str, records: list):
#     all_results = []
#     # Wrap your list in tqdm()
#     for record in tqdm(records, desc="Extracting Genes via LLM"):
#
#         for record in records:
#             title = record["title"]
#             abstract = record["abstract"]
#             pmid = record["pmid"]
#
#             if not abstract:
#                 continue
#
#             prompt = f"""
#             From the following abstract about {disease},
#
#             Identify genes whose expression levels are altered in patients with {disease}
#             under baseline disease conditions (NOT after drug treatment).
#
#             STRICT RULES:
#             - Extract only genes dysregulated in disease patients compared to controls.
#             - Ignore genes altered due to drug treatment or experimental manipulation.
#             - If gene is upregulated/activated/positively regulated (turning on gene expression) in disease → direction = "UP"
#             - If gene is downregulated/suppressed/negatively regulated (turning off gene expression) in disease  → direction = "DOWN"
#             - evidence_type must be:
#                 "BASELINE" or "INTERVENTION"
#
#             Return structured output only.
#             """
#
#             try:
#                 result = structured_llm.invoke(prompt + "\n\n" + abstract)
#
#                 for item in result.root:
#                     gene_data = item.model_dump(mode="json")
#                     gene_data["paper_title"] = title
#                     gene_data["pmid"] = pmid
#                     all_results.append(gene_data)
#
#             except Exception as e:
#                 print(f"Extraction failed for PMID {pmid}:", e)
#
#     return all_results
#
# if __name__ == "__main__":
#
#     disease = "Parkinson's disease"
#
#     print("\n🔎 Fetching PubMed records...")
#     records = fetch_pubmed_records(disease, max_results=5)
#
#     print(f"Fetched {len(records)} records.")
#
#     print("\n🧠 Extracting gene dysregulation per paper...")
#     extracted = extract_baseline_dysregulation(disease, records)
#
#     print("\n📦 Raw Extracted with Paper Info:")
#     for item in extracted:
#         print(item)
#
#     baseline_only = filter_baseline_only(extracted)
#
#     print("\n✅ Baseline Dysregulated Genes Only (with source paper):")
#     for item in baseline_only:
#         print(item)

######################################################################################################################
