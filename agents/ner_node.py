#ner_node.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from shared.schemas import AgentState
from shared.helpers import process_abstract_batch, validate_gene


# ======================================================
# 🟩 NER Node  (Hybrid: PubTator3 + LLM)
#
# Flow per abstract:
#   1. PubTator3  → extracts gene names (fast, rule-based, free)
#   2. LLM        → classifies UP/DOWN + BASELINE/INTERVENTION
#   3. Fallback   → if PubTator3 finds nothing, LLM does full NER
#
# NOTE: Because PubTator3 is called per-abstract (not per-batch),
# batching here groups abstracts to reduce LLM call overhead.
# PubTator3 calls happen INSIDE process_abstract_batch, one per abstract.
# ======================================================

def ner_node(state: AgentState) -> AgentState:

    disease       = state["disease"]
    abstract_list = state.get("abstracts", [])

    if not abstract_list:
        print("⚠️ [NER Node] No abstracts to process.")
        return {**state, "disease_genes": []}

    # Smaller batch size than before — each abstract now also makes
    # a PubTator3 HTTP call, so we keep batches small to avoid
    # overloading the executor and hitting rate limits.
    BATCH_SIZE = 3

    # Split abstracts into batches
    batches = [
        abstract_list[i:i + BATCH_SIZE]
        for i in range(0, len(abstract_list), BATCH_SIZE)
    ]

    print(f"\n🧬 [NER Node] Hybrid mode: PubTator3 + LLM")
    print(f"   Processing {len(abstract_list)} abstracts in {len(batches)} batches of {BATCH_SIZE}...")

    args_list = [(disease, batch) for batch in batches]

    all_extracted = []

    # Parallel over batches
    # max_workers kept at 5 (was 10) — PubTator3 rate limit is ~3 req/sec
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_abstract_batch, args): args for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            result = future.result()
            if result:
                all_extracted.extend(result)

    # ── Provenance summary ──────────────────────────────────────────
    pubtator_count = sum(1 for i in all_extracted if i.get("extraction_source") == "pubtator+llm")
    llm_only_count = sum(1 for i in all_extracted if i.get("extraction_source") == "llm_only")
    print(f"\n📊 Extraction sources — PubTator3+LLM: {pubtator_count} | LLM-only fallback: {llm_only_count}")

    # Filter baseline only
    baseline_only = [i for i in all_extracted if i.get("evidence_type") == "BASELINE"]
    print(f"🔬 Baseline entries after filter: {len(baseline_only)}")

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
