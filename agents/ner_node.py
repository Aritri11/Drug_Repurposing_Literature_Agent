#ner_node.py
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from shared.schemas import AgentState
from shared.helpers import process_abstract_batch, validate_gene
from shared.config import llm_ner


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

CONSECUTIVE_TIMEOUT_THRESHOLD = 3
RECOVERY_WAIT                  = 30


def check_ollama_status():
    """
    Query Ollama's /api/ps endpoint to see what models are
    currently loaded in VRAM. Prints a diagnostic line.
    """
    try:
        r      = requests.get("http://localhost:11434/api/ps", timeout=5)
        models = r.json().get("models", [])
        if not models:
            print("   🔴 Ollama diagnostic: NO models loaded in VRAM right now — model was unloaded!")
        else:
            for m in models:
                name       = m.get("name", "unknown")
                expires_at = m.get("expires_at", "unknown")
                size_vram  = m.get("size_vram", 0)
                print(f"   🟢 Ollama diagnostic: '{name}' is loaded | VRAM: {size_vram//1024//1024}MB | expires: {expires_at}")
    except Exception as e:
        print(f"   ⚠️  Ollama diagnostic: could not reach Ollama API — {e}")


def _ping_ollama() -> bool:
    """Send a tiny warmup ping to Ollama to trigger model reload."""
    try:
        llm_ner.invoke("Say OK")
        return True
    except Exception:
        return False


def ner_node(state: AgentState) -> AgentState:

    disease       = state["disease"]
    abstract_list = state.get("abstracts", [])

    if not abstract_list:
        print("⚠️ [NER Node] No abstracts to process.")
        return {**state, "disease_genes": []}

    BATCH_SIZE = 3

    batches = [
        abstract_list[i:i + BATCH_SIZE]
        for i in range(0, len(abstract_list), BATCH_SIZE)
    ]

    print(f"\n🧬 [NER Node] Hybrid mode: PubTator3 + LLM (sequential batches, parallel within batch)")
    print(f"   Processing {len(abstract_list)} abstracts in {len(batches)} batches of {BATCH_SIZE}...")

    all_extracted     = []
    consecutive_empty = 0

    for batch in tqdm(batches, desc="Processing batches"):
        args                  = (disease, batch)
        result, timeout_count = process_abstract_batch(args)  # unpack tuple

        if result:
            all_extracted.extend(result)
            consecutive_empty = 0
        else:
            consecutive_empty += 1

        # ── Circuit breaker ────────────────────────────────────────────
        if consecutive_empty >= CONSECUTIVE_TIMEOUT_THRESHOLD:
            print(f"\n⚠️  [{consecutive_empty} consecutive empty batches detected]")

            # ← DIAGNOSTIC: check what Ollama actually has loaded right now
            check_ollama_status()

            print(f"   Waiting {RECOVERY_WAIT}s and pinging to reload...")
            time.sleep(RECOVERY_WAIT)

            if _ping_ollama():
                print("   ✅ Ollama responded — resuming.")
            else:
                print("   ⚠️  Ollama still not responding — continuing anyway.")

            consecutive_empty = 0

    # ── Provenance summary ──────────────────────────────────────────────
    pubtator_count = sum(1 for i in all_extracted if i.get("extraction_source") == "pubtator+llm")
    llm_only_count = sum(1 for i in all_extracted if i.get("extraction_source") == "llm_only")
    print(f"\n📊 Extraction sources — PubTator3+LLM: {pubtator_count} | LLM-only fallback: {llm_only_count}")

    # Filter baseline only
    baseline_only = [i for i in all_extracted if i.get("evidence_type") == "BASELINE"]
    print(f"🔬 Baseline entries after filter: {len(baseline_only)}")

    # Validate gene symbols in parallel (MyGene is external API — parallel is fine)
    validated = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(validate_gene, item) for item in baseline_only]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating genes"):
            result = future.result()
            if result:
                validated.append(result)

    print(f"✅ Extracted {len(validated)} validated baseline genes.")
    return {**state, "disease_genes": validated}
