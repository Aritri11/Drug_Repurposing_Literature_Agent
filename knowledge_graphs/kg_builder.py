from knowledge_graphs.neo4j_client import  run_query

def build_kg(state):
    entities = state.get("entities", [])
    disease = state.get("disease")

    for item in entities:
        for d in item["drugs"]:
            for g in item["genes"]:
                for ds in item["diseases"]:

                    q = """
                    MERGE (dr:Drug {name:$d})
                    MERGE (g:Gene {name:$g})
                    MERGE (di:Disease {name:$di})

                    MERGE (dr)-[:TARGETS]->(g)
                    MERGE (g)-[:ASSOCIATED_WITH]->(di)
                    """

                    run_query(q, {
                        "d": d,
                        "g": g,
                        "di": ds
                    })

    return {
        "kg_done": True,
        "disease": disease
    }
