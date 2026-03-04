#This code takes the gene information from the baseline_dysregulation_extractor.py for a particular disease and then hits the DGIdb API
#to see if any drug information exists for those genes and returns that info along with the interaction type

from baseline_dysregulation_extractor import run_extraction_pipeline
from dgidb_extracts import fetch_dgidb_interactions


def get_drug_gene_candidates(disease: str, max_results: int = 20):

    # Step 1: Extract genes from PubMed
    extracted_genes = run_extraction_pipeline(disease, max_results)

    results = []

    for item in extracted_genes:

        gene = item["gene"]
        direction = item["direction"]

        print(f"\n🔎 Checking DGIdb for gene: {gene}")

        interactions = fetch_dgidb_interactions(gene)

        if not interactions:
            print("No drugs found.")
            continue

        for interaction in interactions:

            drug = interaction["drug"]["name"]

            interaction_types = [
                t["type"] for t in interaction.get("interactionTypes", [])
            ]

            results.append({
                "drug": drug,
                "gene": gene,
                "gene_direction": direction,
                "interaction_type": interaction_types
            })

    return results

if __name__ == "__main__":

    disease = "Parkinson's disease"

    candidates = get_drug_gene_candidates(disease, max_results=20)

    print("\nFinal Drug-Gene Candidates:\n")

    for c in candidates:
        print(c)