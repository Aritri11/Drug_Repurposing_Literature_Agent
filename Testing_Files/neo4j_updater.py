"""
Neo4j Updater Module (Fully Dynamic)

This file:
1. Calls baseline_dysregulation_extractor dynamically
2. Clears old relationships
3. Inserts new ones
4. Ensures no duplicate nodes
"""

from test_file import graph
from baseline_dysregulation_extractor import run_extraction_pipeline


# ==========================================================
# 🧱 Create Constraints
# ==========================================================

def create_constraints():
    constraint_queries = [
        """
        CREATE CONSTRAINT disease_name IF NOT EXISTS
        FOR (d:Disease) REQUIRE d.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT gene_symbol IF NOT EXISTS
        FOR (g:Gene) REQUIRE g.symbol IS UNIQUE
        """
    ]

    driver = graph.get_driver()

    with driver.session() as session:
        for query in constraint_queries:
            session.run(query)

    driver.close()

    print("✅ Constraints ensured.")


# ==========================================================
# 🧹 Reset Disease Relationships
# ==========================================================

def reset_disease_relationships(disease: str):

    query = """
    MATCH (d:Disease {name:$disease})-[r:ALTERS_EXPRESSION]->()
    DELETE r
    """

    driver = graph.get_driver()

    with driver.session() as session:
        session.run(query, disease=disease)

    driver.close()

    print(f"🧹 Old relationships cleared for {disease}.")


# ==========================================================
# 🧬 Update Graph Dynamically
# ==========================================================

def update_graph_for_disease(disease: str, max_results: int = 20):

    print(f"\n🔎 Running baseline extraction for {disease}...")

    extracted_data = run_extraction_pipeline(disease, max_results)

    if not extracted_data:
        print("⚠️ No baseline genes extracted.")
        return

    # Clean previous relationships
    reset_disease_relationships(disease)

    query = """
    MERGE (d:Disease {name:$disease})
    MERGE (g:Gene {symbol:$gene})

    MERGE (d)-[r:ALTERS_EXPRESSION]->(g)
    SET r.direction = $direction,
        r.evidence_count = 1,
        r.confidence_score = 0.1
    """

    inserted_count = 0

    driver = graph.get_driver()

    with driver.session() as session:
        for item in extracted_data:

            if item.get("evidence_type") != "BASELINE":
                continue

            session.run(
                query,
                disease=disease,
                gene=item["gene"],
                direction=item["direction"]
            )

    driver.close()


# ==========================================================
# 🚀 Main Entry
# ==========================================================

if __name__ == "__main__":

    disease_name = "Parkinson's disease"

    create_constraints()
    update_graph_for_disease(disease_name, max_results=20)