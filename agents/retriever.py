from Bio import Entrez
from project_config import EMAIL

Entrez.email = EMAIL

def retrieve(state):
    disease = state["disease"]

    handle = Entrez.esearch(
        db="pubmed",
        term=f"{disease} drug treatment",
        retmax=5
    )

    ids = Entrez.read(handle)["IdList"]

    docs = []
    for pmid in ids:
        f = Entrez.efetch(db="pubmed", id=pmid,
                          rettype="abstract", retmode="text")
        docs.append(f.read())

    return {
        "docs": docs,
        "disease": disease
    }
