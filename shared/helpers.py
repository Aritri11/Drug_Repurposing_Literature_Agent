#helpers.py
import requests
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import mygene

import re
import nltk
from nltk.tokenize import sent_tokenize

from shared.schemas import structured_llm
from shared.config import llm_ner as _llm_ner  # used to read model name only
_NER_MODEL = _llm_ner.model  # e.g. "llama3.1:8b" — read once at import time

import threading
_OLLAMA_RESET_LOCK = threading.Lock()
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


# ======================================================
# 🔧 Helper: Reset Ollama model after a hung request
#
# When an LLM call times out, Ollama is still processing
# the hung request internally. Any new request queues behind
# it and also times out. The fix is to:
#   1. Unload the model (keep_alive=0) — kills the hung request
#   2. Reload the model (keep_alive=-1) — fresh start, no queue
#
# This takes ~5-10s but prevents cascading timeouts on all
# subsequent abstracts after a single hung one.
# ======================================================

def _reset_ollama_model(model: str = None, wait_before_reload: float = 3.0):
    """
    Force-reset Ollama model to clear stuck/hung generations:
      1) unload model   (keep_alive=0)
      2) short wait
      3) reload model   (tiny prompt, keep_alive=-1)
    """
    if model is None:
        model = _NER_MODEL

    try:
        import requests as _req

        print(f"             🔄 Resetting Ollama model: {model}")

        # 1) Unload model (kills queued/hung work for this model)
        try:
            r_unload = _req.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": "",        # empty prompt is fine for keep_alive control
                    "stream": False,
                    "keep_alive": 0
                },
                timeout=20
            )
            print(f"             📤 Unload status: {r_unload.status_code}")
        except Exception as e:
            print(f"             ⚠️  Unload request failed: {e}")

        # 2) brief pause so Ollama can finalize unload
        time.sleep(wait_before_reload)

        # 3) Reload + warm model in memory
        try:
            r_reload = _req.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": "OK",
                    "stream": False,
                    "keep_alive": -1
                },
                timeout=60
            )
            if r_reload.status_code == 200:
                print("             ✅ Ollama model reloaded and responsive")
            else:
                print(f"             ⚠️  Reload returned status {r_reload.status_code}")
        except Exception as e:
            print(f"             ⚠️  Reload request failed: {e}")

    except Exception as e:
        print(f"             ⚠️  Ollama reset failed (non-fatal): {e}")

# ── Keywords that signal an abstract is worth processing ─────────────────────
# Abstract must contain at least one expression-change word to be worth
# sending to PubTator3 or the LLM. Abstracts about drug mechanisms,
# reviews, or gene therapy with no dysregulation language are skipped early.
_DYSREGULATION_KEYWORDS = {
    "upregulated", "downregulated", "overexpressed", "underexpressed",
    "increased expression", "decreased expression", "elevated", "reduced",
    "suppressed", "activated", "inhibited", "silenced", "knocked down",
    "knockdown", "upregulation", "downregulation", "overexpression",
    "dysregulated", "dysregulation", "differentially expressed",
    "higher expression", "lower expression", "gene expression",
}

def _abstract_has_dysregulation(abstract: str) -> bool:
    """
    Fast keyword pre-filter — returns True if the abstract contains
    at least one expression-change term. Skips PubTator3 + LLM calls
    for abstracts that clearly have no dysregulation content.
    """
    text = abstract.lower()
    return any(kw in text for kw in _DYSREGULATION_KEYWORDS)


# ======================================================
# 🔧 Helper: Extract relevant sentences (token reduction)
# ======================================================

def extract_relevant_sentences(abstract: str, genes: list[str] = None, max_sentences: int = 5) -> str:
    """
    Extract only sentences with gene mentions + dysregulation keywords.
    Reduces token count sent to LLM by 60–80%.

    Args:
        abstract: Full abstract text
        genes: List of gene names to look for (whitelist)
        max_sentences: Cap output at this many sentences

    Returns:
        Compact string of relevant sentences only
    """
    sentences = sent_tokenize(abstract)

    if not sentences:
        return abstract  # Fallback if tokenization fails

    dysregulation_terms = {
        "upregulated", "downregulated", "overexpressed", "underexpressed",
        "increased", "decreased", "elevated", "reduced", "suppressed",
        "activated", "inhibited", "silenced", "knocked down", "dysregulated",
        "higher expression", "lower expression"
    }

    relevant = []
    gene_pattern = "|".join(re.escape(g) for g in (genes or [])) if genes else ""

    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()

        # Keep if: has dysregulation term OR mentions a gene from whitelist
        has_dys = any(term in sent_lower for term in dysregulation_terms)
        has_gene = gene_pattern and re.search(gene_pattern, sent, re.IGNORECASE)

        if has_dys or has_gene:
            relevant.append(sent)

    if not relevant:
        # Fallback: return first 3 sentences if nothing matched filter
        return " ".join(sentences[:3])

    # Cap at max_sentences and rejoin
    return " ".join(relevant[:max_sentences])



def process_abstract_batch(args: tuple) -> list:
    """
    Hybrid processing for a batch of abstracts.

    STEP 0 — Pre-filter: skip abstracts with no dysregulation keywords
    STEP 1 — One PubTator3 call for the relevant PMIDs in this batch
    STEP 2 — Per abstract: LLM classifies genes found by PubTator3
              OR full LLM NER if PubTator3 had no annotations for that PMID
    """
    disease, batch = args  # batch = [{"pmid": ..., "abstract": ...}, ...]

    # ── STEP 0: Pre-filter — skip abstracts with no dysregulation language ──
    relevant  = []
    skipped   = []
    for item in batch:
        if _abstract_has_dysregulation(item["abstract"]):
            relevant.append(item)
        else:
            skipped.append(item["pmid"])

    if skipped:
        print(f"  ⏭️  Skipped {len(skipped)} abstract(s) — no dysregulation keywords: {', '.join(skipped)}")

    if not relevant:
        return []  # entire batch filtered — no PubTator3 or LLM calls needed

    # ── STEP 1: One PubTator3 call for relevant PMIDs only ──────────────────
    pmids        = [item["pmid"] for item in relevant]
    pubtator_map = fetch_pubtator3_genes_batch(pmids)

    # ── STEP 2: Build prompts for all abstracts in this batch ────────────────
    tasks = []
    for item in relevant:
        pmid           = item["pmid"]
        abstract       = item["abstract"]
        pubtator_genes = pubtator_map.get(pmid, [])

        if pubtator_genes:
            abstract_condensed = extract_relevant_sentences(abstract, pubtator_genes, max_sentences=5)
            gene_list_str = ", ".join(pubtator_genes)
            log_prefix = (
                "  🔬 PMID " + pmid + " | 🟢 PubTator3 → " + str(len(pubtator_genes)) + " gene(s): " + gene_list_str + " → 🤖 LLM: classify UP/DOWN + BASELINE/INTERVENTION only"
            )
            prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
                disease=disease, gene_list=gene_list_str, pmid=pmid, abstract=abstract_condensed
            )
            source = "pubtator+llm"
        else:
            abstract_condensed = extract_relevant_sentences(abstract, max_sentences=5)
            log_prefix = "  🔬 PMID " + pmid + " | 🟡 PubTator3: not indexed → 🤖 LLM: full NER + classification"
            prompt = FALLBACK_NER_PROMPT_TEMPLATE.format(
                disease=disease, pmid=pmid, abstract=abstract_condensed
            )
            source = "llm_only"

        tasks.append((pmid, prompt, source, log_prefix))

    # ── STEP 3: LLM calls in parallel within this batch ──────────────────────
    # Each call is fully independent — safe to parallelise.
    # max_workers = batch size (typically 3) — never more than BATCH_SIZE concurrent calls.
    # The next batch in ner_node.py starts only after this entire block resolves.
    MAX_LLM_RETRIES = 3
    LLM_RETRY_DELAY = 5.0

    def invoke_one(task):
        t_pmid, t_prompt, t_source, t_log_prefix = task
        log = [t_log_prefix]
        result = None
        status = "unknown"  # success | empty | timeout | error

        for attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                # Fresh executor per attempt — thread is fully killed after each call.
                # shutdown(wait=True) blocks until the worker thread exits cleanly,
                # so no lingering threads carry over to the next request.
                _ex  = ThreadPoolExecutor(max_workers=1)
                _fut = _ex.submit(structured_llm.invoke, t_prompt)
                _timed_out = False
                try:
                    result = _fut.result(timeout=LLM_CALL_TIMEOUT)
                except FuturesTimeoutError:
                    _timed_out = True
                    raise  # re-raise so the outer except catches it
                finally:
                    _fut.cancel()  # cancel if still pending (no-op if already done)
                    if _timed_out:
                        # Thread is hung — don't block waiting for it.
                        # Abandon it and let the OS clean up eventually.
                        _ex.shutdown(wait=False)
                    else:
                        # Thread finished cleanly — wait=True guarantees it is
                        # fully dead before the next request starts.
                        _ex.shutdown(wait=True)

                if result and getattr(result, "root", None):
                    status = "success"
                else:
                    status = "empty"
                break

            except FuturesTimeoutError:
                status = "timeout"
                if attempt < MAX_LLM_RETRIES:
                    log.append(
                        f"             ⏱️ LLM timed out after {LLM_CALL_TIMEOUT}s "
                        f"(attempt {attempt}/{MAX_LLM_RETRIES}) — retrying..."
                    )
                    time.sleep(LLM_RETRY_DELAY)
                else:
                    log.append(f"             ⏱️ LLM timed out {MAX_LLM_RETRIES}x — skipping this abstract")
                    # ── Ollama reset on timeout ───────────────────────────
                    # The hung request is still being processed by Ollama
                    # internally. If we don't reset, the next abstract's
                    # request queues behind the hung one and also times out.
                    # Solution: forcibly restart the model via Ollama API
                    # so it starts fresh with no hung requests in its queue.
                    with _OLLAMA_RESET_LOCK:
                        _reset_ollama_model()

            except Exception as err:
                err_str = str(err)
                is_cuda = "CUDA error" in err_str or "status code: 500" in err_str
                is_ratelim = "status code: 429" in err_str or "too many concurrent" in err_str.lower()

                if (is_ratelim or is_cuda) and attempt < MAX_LLM_RETRIES:
                    wait = LLM_RETRY_DELAY * (attempt * 3 if is_ratelim else 1)
                    tag = "🚦 Rate limited (429)" if is_ratelim else "⚠️ Ollama CUDA error"
                    log.append(
                        f"             {tag} (attempt {attempt}/{MAX_LLM_RETRIES}) — waiting {int(wait)}s..."
                    )
                    time.sleep(wait)
                    continue

                status = "error"
                log.append(f"             ❌ Failed after {attempt} attempt(s): {err_str[:120]}")
                result = None
                break

        if status == "success":
            kept = [(e.gene, e.direction.value, e.evidence_type.value) for e in result.root]
            kept_str = ", ".join([f"{g}({d},{et})" for g, d, et in kept])
            log.append(f"             ✅ Kept {len(kept)}: {kept_str}")
        elif status == "empty":
            log.append("             ⚪ LLM completed — no clearly dysregulated genes found")
        elif status == "timeout":
            log.append("             ⏭️ Skipped due to repeated LLM timeout")
        else:
            log.append("             ⏭️ Skipped due to LLM error")

        return t_pmid, result if status in ("success", "empty") else None, t_source, log, status

    # ── Run LLM calls in parallel with staggered submission ────────────────────
    # Problem with simultaneous parallel calls to Ollama:
    #   All 3 fire at t=0 → Ollama queues requests 2 & 3 → they waste queue
    #   time while the 90s timeout clock runs → false timeouts.
    #
    # Solution: stagger submissions by STAGGER_DELAY seconds so Ollama has
    # time to start processing each request before the next one arrives:
    #   t=0s:  Request 1 → Ollama starts immediately
    #   t=5s:  Request 2 → Ollama starts (request 1 is ~halfway done)
    #   t=10s: Request 3 → Ollama starts (request 1 nearly done)
    #   Each request gets ~80s of actual GPU time within the 90s timeout.
    STAGGER_DELAY = 5.0  # seconds — tune based on your avg LLM response time

    all_extracted = []
    batch_results = {}
    futures_list  = []

    with ThreadPoolExecutor(max_workers=len(tasks)) as batch_ex:
        for i, task in enumerate(tasks):
            if i > 0:
                time.sleep(STAGGER_DELAY)  # stagger submissions
            fut = batch_ex.submit(invoke_one, task)
            futures_list.append((task[0], fut))  # (pmid, future)

        # Wait for all futures in this batch to complete
        for pmid_key, fut in futures_list:
            pmid_r, result_r, source_r, log_r, status_r = fut.result()
            batch_results[pmid_r] = (result_r, source_r, log_r, status_r)

    # Print logs in original PMID order for clean output
    for task in tasks:
        pmid = task[0]
        result_r, source_r, log_r, status_r = batch_results[pmid]
        print("\n".join(log_r))
        if result_r and result_r.root:
            for entry in result_r.root:
                data = entry.model_dump(mode="json")
                data["extraction_source"] = source_r
                all_extracted.append(data)

    # Count how many abstracts timed out in this batch
    timeout_count = sum(
        1 for pmid_k in batch_results
        if batch_results[pmid_k][3] == "timeout"  # index 3 = status
    )
    return all_extracted, timeout_count
