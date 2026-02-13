from transformers import pipeline

rebel = pipeline(
    "text2text-generation",
    model="Babelscape/rebel-large"
)

def extract_relations(state):
    docs = state["docs"]

    triples = []

    for d in docs[:5]:
        out = rebel(d[:1024], max_length=256)[0]["generated_text"]
        triples.append(out)

    return {"triples_raw": triples}
