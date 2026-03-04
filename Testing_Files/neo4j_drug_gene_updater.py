"""
Neo4j Drug-Gene Updater

This module:
1. Runs DGIdb connector
2. Inserts Drug-Gene interactions into Neo4j
"""

from graph_backend import Neo4jGraph
from dgidb_connector import get_drug_gene_candidates
from test_file import graph


# ------------------------------------------------
# Create Constraints
# ------------------------------------------------

def create_constraints():

    constraint_queries = [

        """
        CREATE CONSTRAINT drug_name IF NOT EXISTS
        FOR (d:Drug) REQUIRE d.name IS UNIQUE
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

    print("✅ Drug/Gene constraints ensured.")


# ------------------------------------------------
# Insert Drug-Gene Relationships
# ------------------------------------------------

def update_drug_gene_graph(disease: str, max_results: int = 20):

    print(f"\n🔎 Fetching DGIdb drug interactions for {disease}")

    candidates = get_drug_gene_candidates(disease, max_results)

    if not candidates:
        print("⚠️ No drug-gene interactions found.")
        return

    query = """
    MERGE (dr:Drug {name:$drug})
    MERGE (g:Gene {symbol:$gene})

    MERGE (dr)-[r:TARGETS]->(g)
    SET r.interaction_type = $interaction
    """

    driver = graph.get_driver()

    inserted = 0

    with driver.session() as session:

        for item in candidates:

            drug = item["drug"]
            gene = item["gene"]

            interaction = ",".join(item["interaction_type"])

            session.run(
                query,
                drug=drug,
                gene=gene,
                interaction=interaction
            )

            inserted += 1

    driver.close()

    print(f"✅ Inserted {inserted} Drug-Gene interactions.")


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    disease_name = "Parkinson's disease"

    create_constraints()

    update_drug_gene_graph(disease_name, max_results=20)