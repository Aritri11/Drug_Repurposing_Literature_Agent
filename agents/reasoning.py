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
          -[:ASSOCIATED_WITH]->(di:Disease)
    WHERE toLower(di.name) CONTAINS toLower($d)
    RETURN DISTINCT d.name AS drug
    LIMIT 20
    """

    # For demo: static list fallback
    drugs = ["Metformin", "Simvastatin", "Aspirin"]


    prompt = f"""
    You are a biomedical expert in drug repurposing.

    Task:
    Suggest exactly 5 drug repurposing candidates for {disease}.

    Rules:
    - Output ONLY a numbered list
    - Each line = Drug name + short rationale
    - No extra text
    - No repeating disease name
    - Exactly 5 items

    Example format:

    1. DrugName — rationale
    2. DrugName — rationale
    3. DrugName — rationale
    4. DrugName — rationale
    5. DrugName — rationale

    Candidates: {drugs}
    """

    res = llm(prompt, max_length=300)

    return {"result": res[0]["generated_text"]}
