import os
import json
from dotenv import load_dotenv
from Bio import Entrez

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
# 📚 Step 1: Fetch PubMed Abstracts
# --------------------------------------------------
# def fetch_pubmed_abstracts(disease: str, max_results: int = 5):
#     try:
#         search_handle = Entrez.esearch(
#             db="pubmed",
#             term=f"{disease} AND (\"differential expression\" OR upregulated OR downregulated)",
#             retmax=max_results
#         )
#         search_results = Entrez.read(search_handle)
#         search_handle.close()
#
#         id_list = search_results["IdList"]
#
#         if not id_list:
#             return ""
#
#         fetch_handle = Entrez.efetch(
#             db="pubmed",
#             id=id_list,
#             rettype="abstract",
#             retmode="text"
#         )
#
#         abstracts = fetch_handle.read()
#         fetch_handle.close()
#
#         return abstracts
#
#     except Exception as e:
#         print("Error fetching PubMed:", e)
#         return ""

#extensive checking
# import re
#
# def contains_gene_keywords(text):
#     gene_keywords = [
#         "gene",
#         "expression",
#         "upregulated",
#         "downregulated",
#         "mRNA",
#         "protein",
#         "transcription"
#     ]
#
#     text_lower = text.lower()
#
#     for word in gene_keywords:
#         if word in text_lower:
#             return True
#
#     return False
#
# def detect_gene_symbols(text):
#     # Gene symbols are often 2–10 uppercase letters/numbers
#     matches = re.findall(r"\b[A-Z0-9]{2,10}\b", text)
#     return list(set(matches))


#Instead of printing the full text, extract structured metadata.
def fetch_pubmed_abstracts(disease: str, max_results: int = 5):
    search_handle = Entrez.esearch(
        db="pubmed",
        term=f"{disease} AND (\"differential expression\" OR upregulated OR downregulated)",
        retmax=max_results
    )
    search_results = Entrez.read(search_handle)
    search_handle.close()

    id_list = search_results["IdList"]

    print("PMIDs found:", id_list)

    fetch_handle = Entrez.efetch(
        db="pubmed",
        id=id_list,
        rettype="medline",
        retmode="text"
    )

    records = fetch_handle.read()
    fetch_handle.close()

    return records


#test
disease = "Alzheimer's disease"

abstracts = fetch_pubmed_abstracts(disease, max_results=5)

print("\n---- PUBMED ABSTRACTS ----\n")
print(abstracts)

# print("\n---- PUBMED ABSTRACTS ----\n")
# print(abstracts[:1500])   # print only first 1500 chars
#
# print("\nKeyword check:", contains_gene_keywords(abstracts))
#
# print("\nPotential gene symbols:", detect_gene_symbols(abstracts))