from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from shared.schemas import AgentState
from shared.helpers import process_abstract_batch, validate_gene


# ======================================================
# 🟩 NER Node
# Extracts disease-gene relationships from abstracts
# Uses parallel processing for speed
# ======================================================

def ner_node(state: AgentState) -> AgentState:

    disease       = state["disease"]
    abstract_list = state.get("abstracts", [])

    if not abstract_list:
        print("⚠️ [NER Node] No abstracts to process.")
        return {**state, "disease_genes": []}

    BATCH_SIZE = 5  # Tune this — 3 to 5 works well

    # Split abstracts into batches
    batches = [
        abstract_list[i:i + BATCH_SIZE]
        for i in range(0, len(abstract_list), BATCH_SIZE)
    ]

    print(f"\n🧬 [NER Node] Processing {len(abstract_list)} abstracts in {len(batches)} batches of {BATCH_SIZE}...")

    args_list = [(disease, batch) for batch in batches]

    all_extracted = []

    # Now parallel over batches instead of individual abstracts
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_abstract_batch, args): args for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            result = future.result()
            if result:
                all_extracted.extend(result)

    # Filter baseline only
    baseline_only = [i for i in all_extracted if i.get("evidence_type") == "BASELINE"]

    # Validate gene symbols in parallel
    validated = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(validate_gene, item) for item in baseline_only]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating genes"):
            result = future.result()
            if result:
                validated.append(result)

    print(f"✅ Extracted {len(validated)} validated baseline genes.")
    return {**state, "disease_genes": validated}