from transformers import pipeline
from knowledge_graphs.neo4j_client import  run_query

llm = pipeline(
  "text2text-generation",
  model="Babelscape/rebel-large"
)


def reason(state):
    disease = state["disease"]

    q = """
    MATCH (d:Drug)-[:TARGETS]->(:Gene)
          -[:ASSOCIATED_WITH]->(di:Disease {name:$d})
    RETURN DISTINCT d.name AS drug
    LIMIT 20
    """

    # For demo: static list fallback
    drugs = ["Metformin", "Simvastatin", "Aspirin"]

    prompt = f"""
Suggest top 5 repurposing drugs for {disease}.
Candidates: {drugs}
Explain briefly.
"""

    res = llm(prompt, max_length=200)

    return {"result": res[0]["generated_text"]}
