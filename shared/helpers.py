#helpers.py
import requests
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import mygene

from shared.schemas import structured_llm

# How long to wait for a single LLM call before declaring it hung (seconds)
LLM_CALL_TIMEOUT = 90


# ======================================================
# 🔧 Helper: Validate genes with MyGene
# ======================================================

mg = mygene.MyGeneInfo()

def validate_gene(item: dict) -> Optional[dict]:
    """Validate a single gene against MyGene. Returns item with normalized symbol or None."""
    try:
        result = mg.query(item["gene"], species="human", size=1)
        if result["hits"]:
            symbol = result["hits"][0].get("symbol")
            if symbol:
                item["gene"] = symbol
                return item
    except Exception as e:
        print(f"Gene validation failed for {item['gene']}: {e}")
    return None


# ======================================================
# 🔧 Helper: PubTator3 — fetch gene annotations for a BATCH of PMIDs
#
# Key design decisions:
#   - Batches multiple PMIDs in one request (comma-separated) → fewer HTTP calls
#   - 400 errors are silently ignored — they mean the PMID is too new/not yet
#     indexed by PubTator3. This is expected and not an error worth printing.
#   - 429 rate limit → exponential backoff
#   - Returns dict {pmid: [gene, ...]} so caller knows which genes belong where
# ======================================================

def fetch_pubtator3_genes_batch(pmids: list[str], retries: int = 3, delay: float = 2.0) -> dict[str, list[str]]:
    """
    Query PubTator3 for gene annotations across multiple PMIDs in one call.
    Returns {pmid: [gene_name, ...]} for PMIDs that have annotations.
    PMIDs with no annotations (new papers, not yet indexed) are simply absent from the result.
    """
    if not pmids:
        return {}

    pmid_str = ",".join(pmids)
    url      = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids={pmid_str}"

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=20)

            # 400 = PMID(s) not yet indexed by PubTator3 (very new papers) — silent fallback
            if response.status_code == 400:
                return {}

            # 429 = rate limited — wait and retry
            if response.status_code == 429:
                wait = delay * (attempt + 1)
                print(f"⚠️ PubTator3 rate limited. Waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            result: dict[str, list[str]] = {}
            for doc in data.get("PubTator3", []):
                # PubTator3 stores the PMID in the top-level "id" field
                doc_pmid = str(doc.get("id", "")).strip()
                # Use normalized gene symbol (identifier field) when available,
                # fall back to raw text. This avoids sending long names like
                # "tumor necrosis factor-alpha" to the LLM — TNF is enough.
                genes = set()
                for passage in doc.get("passages", []):
                    for annotation in passage.get("annotations", []):
                        infons = annotation.get("infons", {})
                        if infons.get("type") == "Gene":
                            # "identifier" is the NCBI Gene ID — not a symbol.
                            # Prefer the annotation text but deduplicate
                            # case-insensitively so "Furin"/"furin" → one entry.
                            gene_text = annotation.get("text", "").strip()
                            if gene_text:
                                # Keep the capitalised/official-looking form
                                genes.add(gene_text)
                if genes:
                    # Deduplicate case-insensitively: keep shortest unique symbol
                    seen: dict[str, str] = {}
                    for g in genes:
                        key = g.upper()
                        # Prefer the shorter string (e.g. "TNF" over "tumor necrosis factor")
                        if key not in seen or len(g) < len(seen[key]):
                            seen[key] = g
                    result[doc_pmid] = list(seen.values())

            return result

        except requests.exceptions.Timeout:
            print(f"⏱️ PubTator3 timeout (attempt {attempt+1}/{retries}). Retrying...")
            time.sleep(delay)
        except Exception as e:
            print(f"❌ PubTator3 batch fetch failed: {e}")
            break

    return {}  # All retries exhausted — LLM fallback will handle everything


# ======================================================
# 🔧 Helper: DGIdb batch API call
# ======================================================

def fetch_dgidb_interactions_batch(gene_list: list) -> dict:
    if not gene_list:
        return {}

    url = "https://dgidb.org/api/graphql"

    query = """
    {
      genes(names: %s) {
        nodes {
          name
          interactions {
            drug {
              name
              approved
            }
            interactionScore
            interactionTypes {
              type
            }
          }
        }
      }
    }
    """ % str(gene_list).replace("'", '"')

    try:
        response = requests.post(url, json={"query": query}, timeout=30)
        response.raise_for_status()
        data = response.json()

        result = {}
        for gene_node in data.get("data", {}).get("genes", {}).get("nodes", []):
            gene         = gene_node.get("name", "")
            interactions = gene_node.get("interactions", [])
            result[gene] = interactions

        return result

    except Exception as e:
        print(f"DGIdb batch fetch failed: {e}")
        return {}


# ======================================================
# 🔧 Helper: Compute evidence strength from score
# ======================================================

def get_evidence_strength(score: float) -> str:
    """
    Compute evidence strength label based on pre-computed score.

    Score ranges:
        70 - 100  → 🟢 High
        40 - 69   → 🟡 Medium
        0  - 39   → 🔴 Low
    """
    if score >= 70:
        return "🟢 High"
    elif score >= 40:
        return "🟡 Medium"
    else:
        return "🔴 Low"


# ======================================================
# 🔧 Helper: Process a batch of abstracts (Hybrid approach)
#
# STEP 1 — PubTator3: Extract gene names (fast, accurate, no GPU)
# STEP 2 — LLM:       Classify direction (UP/DOWN) and context
#                     (BASELINE/INTERVENTION) for the found genes
#
# If PubTator3 finds no genes for an abstract, we fall back to
# asking the LLM to do full NER + classification for that abstract.
# ======================================================

# --- Prompt when PubTator3 found genes (classification only) ---
# Key design: LLM is given an EXPLICIT WHITELIST of genes from PubTator3.
# It must ONLY output genes from that whitelist that have a CLEAR direction
# stated in the abstract. Genes merely mentioned (not dysregulated) must be omitted.
CLASSIFICATION_PROMPT_TEMPLATE = """You are a strict biomedical classifier. Do NOT perform gene recognition — that is already done.

Your ONLY job: for each gene in the WHITELIST below, decide if the abstract explicitly states it is dysregulated in {disease} patients, and if so, classify it.

WHITELIST (the only genes you may output): {gene_list}

CLASSIFICATION RULES — a gene qualifies ONLY if the abstract contains an explicit statement of:
- Upregulation / overexpression / increased expression / elevated levels / activation → direction = "UP"
- Downregulation / suppression / decreased expression / reduced levels / inhibition → direction = "DOWN"

evidence_type rules:
- "BASELINE" → dysregulation occurs naturally in {disease} patients vs healthy controls (no treatment)
- "INTERVENTION" → dysregulation is caused by a drug, treatment, knockdown, or experimental manipulation

REJECT a gene if:
- It is merely mentioned in the abstract without a clear direction statement
- Its direction is described only after treatment (mark as INTERVENTION, not BASELINE)
- You are uncertain — when in doubt, OMIT

PMID: {pmid}
Abstract: {abstract}

Return structured output only. If no genes qualify, return an empty list."""

# --- Fallback prompt when PubTator3 found nothing (full NER + classification) ---
FALLBACK_NER_PROMPT_TEMPLATE = """You are a strict biomedical NER and classifier for {disease} research.

TASK: Find genes in the abstract that are EXPLICITLY stated as dysregulated in {disease} patients.

EXTRACTION RULES — only extract a gene if the abstract contains:
- Upregulation / overexpression / increased expression / elevated levels → direction = "UP"
- Downregulation / suppression / decreased expression / reduced levels → direction = "DOWN"

evidence_type rules:
- "BASELINE" → change occurs naturally in {disease} patients vs healthy controls (no drug/treatment)
- "INTERVENTION" → change is caused by a drug, treatment, knockdown, or experimental manipulation

REJECT if:
- Gene is mentioned but no expression direction is stated
- You are guessing or inferring — only extract what is explicitly written
- Abstract is about animal models only (not human patients)

PMID: {pmid}
Abstract: {abstract}

Return structured output only. If nothing qualifies, return an empty list."""


def process_abstract_batch(args: tuple) -> list:
    """
    Hybrid processing for a batch of abstracts.

    STEP 1 — One PubTator3 call for the entire batch (all PMIDs at once)
    STEP 2 — Per abstract: LLM classifies genes found by PubTator3
              OR full LLM NER if PubTator3 had no annotations for that PMID
    """
    disease, batch = args  # batch = [{"pmid": ..., "abstract": ...}, ...]

    # ── STEP 1: One PubTator3 call for all PMIDs in this batch ─────
    pmids          = [item["pmid"] for item in batch]
    pubtator_map   = fetch_pubtator3_genes_batch(pmids)  # {pmid: [genes]}

    all_extracted = []

    for item in batch:
        pmid     = item["pmid"]
        abstract = item["abstract"]
        log      = []  # Buffer all log lines — print atomically to avoid thread interleaving

        try:
            pubtator_genes = pubtator_map.get(pmid, [])

            # ── STEP 2a: PubTator3 found genes → LLM classifies only ──
            if pubtator_genes:
                gene_list_str = ", ".join(pubtator_genes)
                log.append(f"  🔬 PMID {pmid} | 🟢 PubTator3 → {len(pubtator_genes)} gene(s): {gene_list_str}")
                log.append(f"             → 🤖 LLM: classify UP/DOWN + BASELINE/INTERVENTION only")
                prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
                    disease   = disease,
                    gene_list = gene_list_str,
                    pmid      = pmid,
                    abstract  = abstract
                )
                source = "pubtator+llm"

            # ── STEP 2b: No PubTator3 annotations → full LLM NER ──────
            else:
                log.append(f"  🔬 PMID {pmid} | 🟡 PubTator3: not indexed → 🤖 LLM: full NER + classification")
                prompt = FALLBACK_NER_PROMPT_TEMPLATE.format(
                    disease  = disease,
                    pmid     = pmid,
                    abstract = abstract
                )
                source = "llm_only"

            # ── STEP 3: LLM call with timeout + retry ──────────────────────
            # LangChain's invoke() has no built-in timeout — a hung Ollama
            # request will block forever. We run it in a thread and enforce
            # LLM_CALL_TIMEOUT seconds. On timeout or CUDA error we retry
            # up to MAX_LLM_RETRIES times with a short backoff.
            MAX_LLM_RETRIES = 3
            LLM_RETRY_DELAY = 5.0
            result = None

            for llm_attempt in range(1, MAX_LLM_RETRIES + 1):
                try:
                    # Submit to executor WITHOUT context manager —
                    # 'with ThreadPoolExecutor' blocks on __exit__ even after
                    # timeout fires, which defeats the timeout entirely.
                    # shutdown(wait=False) lets us abandon the hung thread.
                    _executor = ThreadPoolExecutor(max_workers=1)
                    _future   = _executor.submit(structured_llm.invoke, prompt)
                    _executor.shutdown(wait=False)
                    result    = _future.result(timeout=LLM_CALL_TIMEOUT)
                    break  # success
                except FuturesTimeoutError:
                    is_last = llm_attempt == MAX_LLM_RETRIES
                    if not is_last:
                        log.append(f"             ⏱️ LLM timed out after {LLM_CALL_TIMEOUT}s (attempt {llm_attempt}/{MAX_LLM_RETRIES}) — retrying...")
                        time.sleep(LLM_RETRY_DELAY)
                    else:
                        log.append(f"             ⏱️ LLM timed out {MAX_LLM_RETRIES}x — skipping this abstract")
                        result = None
                        break
                except Exception as llm_err:
                    err_str = str(llm_err)
                    is_cuda = "CUDA error" in err_str or "status code: 500" in err_str
                    is_last = llm_attempt == MAX_LLM_RETRIES

                    if is_cuda and not is_last:
                        log.append(f"             ⚠️ Ollama CUDA error (attempt {llm_attempt}/{MAX_LLM_RETRIES}) — waiting {LLM_RETRY_DELAY}s...")
                        time.sleep(LLM_RETRY_DELAY)
                    else:
                        raise

            if result and result.root:
                kept = [(e.gene, e.direction.value, e.evidence_type.value) for e in result.root]
                kept_str = ", ".join([f"{g}({d},{et})" for g, d, et in kept])
                log.append(f"             ✅ Kept {len(kept)}: {kept_str}")
                for entry in result.root:
                    data = entry.model_dump(mode="json")
                    data["extraction_source"] = source
                    all_extracted.append(data)
            else:
                log.append(f"             ⚪ No clearly dysregulated genes found")

        except Exception as e:
            log.append(f"             ❌ Failed: {e}")

        finally:
            # Print all lines for this abstract at once — avoids interleaved output
            print("\n".join(log))

    return all_extracted
