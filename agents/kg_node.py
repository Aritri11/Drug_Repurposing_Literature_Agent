from shared.schemas import AgentState
from shared.config import get_neo4j_driver


# ======================================================
# 🟥 Knowledge Graph Node
# Writes Disease→Gene and Drug→Gene relationships to Neo4j
# ======================================================

def kg_node(state: AgentState) -> AgentState:

    disease       = state["disease"]
    disease_genes = state.get("disease_genes", [])
    drug_cands    = state.get("drug_candidates", [])

    print(f"\n🗄️  [KG Node] Writing to Neo4j...")

    driver = get_neo4j_driver()

    try:
        with driver.session() as session:

            # --- Constraints ---
            session.run("CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE")
            session.run("CREATE CONSTRAINT gene_symbol IF NOT EXISTS FOR (g:Gene) REQUIRE g.symbol IS UNIQUE")
            session.run("CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE")

            # Step 1 — Delete old Disease→Gene
            session.run(
                "MATCH (d:Disease {name:$disease})-[r:ALTERS_EXPRESSION]->() DELETE r",
                disease=disease
            )

            # Step 2 — Delete old Drug→Gene for this disease's genes only
            session.run("""
                MATCH (d:Disease {name:$disease})-[:ALTERS_EXPRESSION]->(g:Gene)<-[r:TARGETS]-()
                DELETE r
            """, disease=disease)

            # Step 3 — Insert Disease→Gene relationships
            disease_gene_query = """
            MERGE (d:Disease {name:$disease})
            MERGE (g:Gene {symbol:$gene})
            MERGE (d)-[r:ALTERS_EXPRESSION]->(g)
            SET r.pmids = CASE
                            WHEN r.pmids IS NULL THEN [$pmid]
                            WHEN NOT $pmid IN r.pmids THEN r.pmids + $pmid
                            ELSE r.pmids
                          END,
                r.evidence_count = CASE
                            WHEN r.pmids IS NULL THEN 1
                            WHEN NOT $pmid IN r.pmids THEN size(r.pmids) + 1
                            ELSE size(r.pmids)
                          END,
                r.confidence_score = CASE
                            WHEN r.pmids IS NULL THEN 0.1
                            WHEN NOT $pmid IN r.pmids THEN (size(r.pmids) + 1) * 0.1
                            ELSE size(r.pmids) * 0.1
                          END,
                r.up_count = CASE
                            WHEN $direction = "UP" THEN coalesce(r.up_count, 0) + 1
                            ELSE coalesce(r.up_count, 0)
                          END,
                r.down_count = CASE
                            WHEN $direction = "DOWN" THEN coalesce(r.down_count, 0) + 1
                            ELSE coalesce(r.down_count, 0)
                          END,
                r.direction = CASE
                            WHEN $direction = "UP" AND coalesce(r.up_count, 0) + 1 >= coalesce(r.down_count, 0) THEN "UP"
                            WHEN $direction = "DOWN" AND coalesce(r.down_count, 0) + 1 > coalesce(r.up_count, 0) THEN "DOWN"
                            ELSE coalesce(r.direction, $direction)
                          END,
                r.conflicted = CASE
                            WHEN coalesce(r.up_count, 0) > 0 AND coalesce(r.down_count, 0) > 0 THEN true
                            ELSE false
                          END
            """

            for item in disease_genes:
                session.run(
                    disease_gene_query,
                    disease=disease,
                    gene=item["gene"],
                    direction=item["direction"],
                    pmid=item["pmid"]
                )

            print(f"  ✅ Wrote {len(disease_genes)} Disease→Gene relationships")

            # Step 4 — Insert Drug→Gene relationships
            drug_gene_query = """
            MERGE (dr:Drug {name:$drug})
            MERGE (g:Gene {symbol:$gene})
            MERGE (dr)-[r:TARGETS]->(g)
            SET r.interaction_type  = $interaction,
                r.interaction_score = $interaction_score,
                r.approved          = $approved
            """

            for item in drug_cands:
                session.run(
                    drug_gene_query,
                    drug=item["drug"],
                    gene=item["gene"],
                    interaction=",".join(item["interaction_type"]),
                    interaction_score=item.get("interaction_score", 0.0),
                    approved=item.get("approved", False)
                )

            print(f"  ✅ Wrote {len(drug_cands)} Drug→Gene relationships")

        return {**state, "kg_status": "done"}

    except Exception as e:
        print(f"❌ Neo4j write failed: {e}")
        return {**state, "kg_status": "error"}

    finally:
        driver.close()