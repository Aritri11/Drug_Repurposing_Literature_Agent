#dgidb_node.py
from shared.schemas import AgentState
from shared.helpers import fetch_dgidb_interactions_batch


# ======================================================
# 🟨 DGIdb Node
# Fetches drug-gene interactions for all extracted genes
# Uses a single batch API call instead of one per gene
# ======================================================

def dgidb_node(state: AgentState) -> AgentState:

    disease_genes = state.get("disease_genes", [])

    if not disease_genes:
        print("⚠️ [DGIdb Node] No genes to query.")
        return {**state, "drug_candidates": []}

    # Get unique gene list
    gene_list = list(set(item["gene"] for item in disease_genes))

    print(f"\n💊 [DGIdb Node] Querying DGIdb for {len(gene_list)} genes in one batch call...")

    # ONE batch API call for all genes
    interactions_map = fetch_dgidb_interactions_batch(gene_list)

    # Build gene direction lookup
    gene_direction_map = {item["gene"]: item["direction"] for item in disease_genes}

    drug_candidates = []

    for gene, interactions in interactions_map.items():
        gene_direction = gene_direction_map.get(gene, "UNKNOWN")

        for interaction in interactions:
            drug      = interaction.get("drug", {}).get("name", "")
            types     = [t["type"] for t in interaction.get("interactionTypes", [])]
            raw_score = interaction.get("interactionScore")
            approved  = interaction.get("drug", {}).get("approved", False)

            if drug:
                drug_candidates.append({
                    "drug":              drug,
                    "gene":              gene,
                    "gene_direction":    gene_direction,
                    "interaction_type":  types,
                    "interaction_score": float(raw_score) if raw_score is not None else 0.0,
                    "approved":          approved
                })

    print(f"✅ Found {len(drug_candidates)} drug-gene candidates.")
    return {**state, "drug_candidates": drug_candidates}
