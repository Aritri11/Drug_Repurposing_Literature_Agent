from Bio import Entrez
from shared.schemas import AgentState


# ======================================================
# 🟦 PubMed Node
# Fetches abstracts from PubMed for the given disease
# ======================================================

def pubmed_node(state: AgentState) -> AgentState:

    disease     = state["disease"]
    max_results = state.get("max_results", 20)

    print(f"\n📚 [PubMed Node] Fetching abstracts for: '{disease}'")

    search_term = f"""
    {disease}[Title/Abstract]
    AND ("expression level" OR "gene expression" OR upregulated OR downregulated OR overexpressed OR underexpressed)
    """

    try:
        search_handle  = Entrez.esearch(db="pubmed", term=search_term, retmax=max_results)
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results["IdList"]

        if not id_list:
            print("⚠️ No PubMed results found.")
            return {**state, "abstracts": []}

        fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml", retmode="xml")
        records      = Entrez.read(fetch_handle)
        fetch_handle.close()

        abstract_list = []
        for article in records["PubmedArticle"]:
            try:
                pmid          = str(article["MedlineCitation"]["PMID"])
                abstract_text = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
                full_abstract = " ".join([str(t) for t in abstract_text])
                abstract_list.append({"pmid": pmid, "abstract": full_abstract})
            except KeyError:
                continue

        print(f"✅ Fetched {len(abstract_list)} abstracts.")
        return {**state, "abstracts": abstract_list}

    except Exception as e:
        print(f"❌ PubMed fetch failed: {e}")
        return {**state, "abstracts": []}