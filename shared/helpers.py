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

# ── NER result cache ──────────────────────────────────────────────────────────
# Persists LLM results to disk so re-running the same disease reuses
# previous results. Keyed by "disease::pmid".
import json as _json, os as _os

_CACHE_PATH  = _os.path.join(_os.path.dirname(__file__), "ner_cache.json")
_NER_CACHE: dict  = {}
_CACHE_DIRTY: bool = False

def _load_ner_cache():
    global _NER_CACHE
    try:
        if _os.path.exists(_CACHE_PATH):
            with open(_CACHE_PATH, "r") as f:
                _NER_CACHE = _json.load(f)
    except Exception:
        _NER_CACHE = {}

def _save_ner_cache():
    try:
        with open(_CACHE_PATH, "w") as f:
            _json.dump(_NER_CACHE, f)
    except Exception as e:
        print(f"⚠️  Cache save failed: {e}")

_load_ner_cache()  # load once at import time

import threading
_OLLAMA_RESET_LOCK = threading.Lock()
# How long to wait for a single LLM call before declaring it hung (seconds)
LLM_CALL_TIMEOUT_BASE = 90  # kept for reference

def _get_timeout_for_genes(gene_count: int) -> float:
    """
    Compute LLM timeout dynamically based on number of genes to classify.
    Each gene = one structured JSON object to generate.
    More genes = longer generation time needed.

        0-5   genes → 90s
        6-10  genes → 120s
        11-15 genes → 150s
        16-20 genes → 200s
        21+   genes → 270s
    """
    if gene_count <= 5:
        return 90.0
    elif gene_count <= 10:
        return 120.0
    elif gene_count <= 15:
        return 150.0
    elif gene_count <= 20:
        return 200.0
    else:
        return 270.0


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

def _reset_ollama_model(model: str = None, wait: float = 20.0):
    """
    Wait for Ollama to clear any hung request, then confirm responsive.
    Does NOT unload/reload — that causes VRAM conflicts when deepseek-r1:8b
    is also loaded, leading to 60s+ reload timeouts.
    """
    if model is None:
        model = _NER_MODEL
    try:
        print(f"             \u23f3 Waiting {wait}s for Ollama queue to clear...")
        time.sleep(wait)
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "keep_alive": -1, "prompt": "", "stream": False},
            timeout=15
        )
        if r.status_code == 200:
            print(f"             \u2705 Ollama responsive \u2014 resuming")
        else:
            print(f"             \u26a0\ufe0f  Ollama ping returned {r.status_code}")
    except Exception as e:
        print(f"             \u26a0\ufe0f  Ollama ping failed (non-fatal): {e}")


# ── Dysregulation keywords (basic check) ──────────────────────────────────────
_DYSREGULATION_KEYWORDS = {
    "upregulated", "downregulated", "overexpressed", "underexpressed",
    "increased expression", "decreased expression", "elevated", "reduced",
    "suppressed", "activated", "inhibited", "silenced", "knocked down",
    "knockdown", "upregulation", "downregulation", "overexpression",
    "dysregulated", "dysregulation", "differentially expressed",
    "higher expression", "lower expression", "gene expression",
}

# ── Explicit dysregulation phrases → near-certain to yield LLM result ─────────
# These are specific enough that if present, LLM almost always finds a gene.
_EXPLICIT_PHRASES = [
    "was upregulated", "were upregulated", "is upregulated",
    "was downregulated", "were downregulated", "is downregulated",
    "was overexpressed", "were overexpressed",
    "was underexpressed", "were underexpressed",
    "significantly upregulated", "significantly downregulated",
    "significantly increased expression", "significantly decreased expression",
    "expression was increased", "expression was decreased",
    "expression was elevated", "expression was reduced",
    "mrna levels were", "protein levels were",
    "mrna expression was", "protein expression was",
    "differentially expressed",
    "increased expression of", "decreased expression of",
    "upregulation of", "downregulation of", "overexpression of",
    "higher expression of", "lower expression of",
]

# ── Abstract types that almost never have baseline gene dysregulation ──────────
_SKIP_ABSTRACT_TYPES = [
    "systematic review", "meta-analysis", "randomized controlled trial",
    "clinical trial", "vaccine efficacy", "pharmacokinetics",
    "drug resistance", "antimicrobial", "case report", "case series",
    "epidemiology", "seroprevalence", "diagnostic accuracy",
]

# ── Direction words for proximity check around gene names ─────────────────────
_DIRECTION_WORDS = {
    "upregulated", "downregulated", "overexpressed", "underexpressed",
    "increased", "decreased", "elevated", "reduced", "suppressed",
    "activated", "inhibited", "upregulation", "downregulation",
    "higher", "lower", "diminished", "enhanced", "overexpression",
}


def _get_signal_strength(abstract: str) -> str:
    """
    Classify abstract signal strength before any API call.

    'strong' -> explicit dysregulation phrase found -> always call LLM
    'weak'   -> generic expression language only -> call LLM only if
                PubTator3 found genes AND proximity check passes
    'none'   -> no dysregulation signal OR irrelevant abstract type -> skip

    This eliminates two common wasted LLM calls:
      Situation 1: PubTator3 not indexed + LLM finds nothing
                   (weak signal abstracts with no genes get skipped)
      Situation 2: PubTator3 found genes + LLM finds nothing
                   (genes merely mentioned, not near direction words)
    """
    text = abstract.lower()

    # Skip known irrelevant abstract types entirely
    if any(skip in text for skip in _SKIP_ABSTRACT_TYPES):
        return "none"

    # Strong: explicit dysregulation phrase present
    if any(phrase in text for phrase in _EXPLICIT_PHRASES):
        return "strong"

    # Weak: only generic expression/regulation language
    if any(kw in text for kw in _DYSREGULATION_KEYWORDS):
        return "weak"

    return "none"


def _gene_has_direction_context(gene_name: str, abstract: str, window: int = 150) -> bool:
    """
    Check if a direction word appears within `window` characters of the
    gene name in the abstract.

    Fixes Situation 2: PubTator3 finds genes like 'GM-Vac, THRIL' that
    are merely mentioned in background/methods — no direction word nearby.
    Returns False -> skip LLM call -> no wasted inference.
    """
    text = abstract.lower()
    gene = gene_name.lower()
    idx  = text.find(gene)

    while idx != -1:
        start   = max(0, idx - window)
        end     = min(len(text), idx + len(gene) + window)
        context = text[start:end]
        if any(dw in context for dw in _DIRECTION_WORDS):
            return True
        idx = text.find(gene, idx + 1)

    return False


def _any_gene_has_direction_context(genes: list, abstract: str) -> bool:
    """Returns True if ANY gene in the list has a direction word nearby."""
    return any(_gene_has_direction_context(g, abstract) for g in genes)


def _abstract_has_dysregulation(abstract: str) -> bool:
    """Legacy helper — kept for backward compatibility."""
    return _get_signal_strength(abstract) != "none"

def extract_relevant_sentences(abstract: str, gene_names: list = None, max_sentences: int = 5) -> str:
    """
    Extract only the sentences most relevant to gene dysregulation.
    Reduces prompt length sent to LLM, speeding up inference.

    Strategy:
    - Keep sentences containing direction words (upregulated, decreased etc.)
    - If gene_names provided, prioritise sentences mentioning those genes
    - Cap at max_sentences to keep prompt short
    - Fall back to full abstract if no relevant sentences found
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
    if not sentences:
        return abstract

    direction_words = {
        "upregulated", "downregulated", "overexpressed", "underexpressed",
        "increased", "decreased", "elevated", "reduced", "suppressed",
        "activated", "inhibited", "upregulation", "downregulation",
        "differentially expressed", "higher expression", "lower expression",
    }

    scored = []
    gene_names_lower = [g.lower() for g in (gene_names or [])]

    for sent in sentences:
        sent_lower = sent.lower()
        score = 0

        # Score: direction word present
        if any(dw in sent_lower for dw in direction_words):
            score += 2

        # Score: gene name mentioned
        if any(g in sent_lower for g in gene_names_lower):
            score += 1

        if score > 0:
            scored.append((score, sent))

    # Sort by score descending, take top max_sentences
    scored.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _, s in scored[:max_sentences]]

    if not top_sentences:
        return abstract  # fallback — return full abstract

    return " ".join(top_sentences)


def process_abstract_batch(args: tuple) -> list:
    """
    Hybrid processing for a batch of abstracts.

    STEP 0 — Pre-filter: skip abstracts with no dysregulation keywords
    STEP 1 — One PubTator3 call for the relevant PMIDs in this batch
    STEP 2 — Per abstract: LLM classifies genes found by PubTator3
              OR full LLM NER if PubTator3 had no annotations for that PMID
    """
    global _NER_CACHE, _CACHE_DIRTY
    disease, batch = args  # batch = [{"pmid": ..., "abstract": ...}, ...]

    # ── STEP 0: Two-stage pre-filter ────────────────────────────────────────
    # Strong signal → always send to LLM
    # Weak signal   → only send if PubTator3 finds genes (confirms relevance)
    # No signal     → skip entirely (~40% of LLM calls eliminated this way)
    relevant = []   # [(item, signal_strength), ...]
    skipped  = []

    for item in batch:
        strength = _get_signal_strength(item["abstract"])
        if strength == "none":
            skipped.append(item["pmid"])
        else:
            relevant.append((item, strength))

    if skipped:
        print(f"  ⏭️  Skipped {len(skipped)} abstract(s) — no dysregulation keywords: {', '.join(skipped)}")

    if not relevant:
        return [], 0

    # ── STEP 1: One PubTator3 call for relevant PMIDs only ──────────────────
    pmids        = [item["pmid"] for item, _ in relevant]
    pubtator_map = fetch_pubtator3_genes_batch(pmids)

    # ── STEP 2: Build prompts for all abstracts in this batch ────────────────
    tasks = []
    for item, signal_strength in relevant:
        pmid           = item["pmid"]
        abstract       = item["abstract"]
        pubtator_genes = pubtator_map.get(pmid, [])

        # ── Stage B: context-aware skip ─────────────────────────────────────
        #
        # Situation 1 — PubTator3 not indexed + weak signal:
        #   No genes found AND no explicit dysregulation phrase
        #   → LLM almost certainly returns empty → skip
        #
        # Situation 2 — PubTator3 found genes but none near a direction word:
        #   Genes are merely mentioned (background/methods), not reported as
        #   dysregulated. Proximity check confirms → skip LLM entirely.
        #
        if not pubtator_genes and signal_strength == "weak":
            print(f"  ⏭️  PMID {pmid} | weak signal + not indexed → skipping LLM")
            continue

        if pubtator_genes and not _any_gene_has_direction_context(pubtator_genes, abstract):
            print(f"  ⏭️  PMID {pmid} | genes found but none near direction words → skipping LLM")
            continue

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

        gene_count = len(pubtator_genes) if pubtator_genes else 0
        tasks.append((pmid, prompt, source, log_prefix, gene_count))

    # ✅ guard for empty tasks
    if not tasks:
        return [], 0

    # ── STEP 3: LLM calls in parallel within this batch ──────────────────────
    # Each call is fully independent — safe to parallelise.
    # max_workers = batch size (typically 3) — never more than BATCH_SIZE concurrent calls.
    # The next batch in ner_node.py starts only after this entire block resolves.
    MAX_LLM_RETRIES = 3
    LLM_RETRY_DELAY = 5.0

    def invoke_one(task):
        t_pmid, t_prompt, t_source, t_log_prefix, t_gene_count = task
        dynamic_timeout = _get_timeout_for_genes(t_gene_count)
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
                    result = _fut.result(timeout=dynamic_timeout)
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
                        f"             ⏱️ LLM timed out after {dynamic_timeout:.0f}s "
                        f"(attempt {attempt}/{MAX_LLM_RETRIES}) — retrying..."
                    )
                    time.sleep(LLM_RETRY_DELAY)
                else:
                    log.append(f"             ⏱️ LLM timed out {MAX_LLM_RETRIES}x (limit={dynamic_timeout:.0f}s) — skipping this abstract")
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
        pmid      = task[0]
        result_r, source_r, log_r, status_r = batch_results[pmid]
        print("\n".join(log_r))
        cache_key_r = disease + "::" + pmid
        if result_r and result_r.root:
            entries = []
            for entry in result_r.root:
                data = entry.model_dump(mode="json")
                data["extraction_source"] = source_r
                entries.append(data)
                all_extracted.append(data)
            _NER_CACHE[cache_key_r] = entries  # cache result
            _CACHE_DIRTY = True
        else:
            _NER_CACHE[cache_key_r] = []       # cache empty result
            _CACHE_DIRTY = True

    # Count how many abstracts timed out in this batch
    timeout_count = sum(
        1 for pmid_k in batch_results
        if batch_results[pmid_k][3] == "timeout"  # index 3 = status
    )
    return all_extracted, timeout_count
