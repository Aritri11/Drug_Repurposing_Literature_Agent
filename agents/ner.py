from transformers import pipeline

ner = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"
)

def extract_entities(state):
    docs = state.get("docs", [])
    disease = state.get("disease")

    entities = []

    for d in docs:
        ents = ner(d[:1000])

        drugs, genes, diseases = [], [], []

        for e in ents:
            label = e["entity_group"].lower()

            if "drug" in label:
                drugs.append(e["word"])

            elif "gene" in label:
                genes.append(e["word"])

            elif "disease" in label:
                diseases.append(e["word"])

        entities.append({
            "drugs": drugs,
            "genes": genes,
            "diseases": diseases
        })

    return {
        "entities": entities,
        "disease": disease
    }
