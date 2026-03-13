import requests
from typing import Optional
import mygene

from shared.schemas import structured_llm


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
# 🔧 Helper: Process single abstract through LLM
# ======================================================

BATCH_EXTRACTION_PROMPT_TEMPLATE = """
From the following PubMed abstracts about {disease},

For EACH abstract, identify genes whose expression levels are altered in patients with {disease}
under baseline disease conditions (NOT after drug treatment).

STRICT RULES:
- Extract only genes dysregulated in disease patients compared to controls.
- Ignore genes altered due to drug treatment or experimental manipulation.
- If gene is upregulated/activated/positively regulated (turning on gene expression) in disease → direction = "UP"
- If gene is downregulated/suppressed/negatively regulated (turning off gene expression) in disease → direction = "DOWN"
- evidence_type = "BASELINE" if naturally dysregulated in disease
- evidence_type = "INTERVENTION" if altered due to treatment
- Use the PMID provided for each abstract exactly as given.

Abstracts:
{abstracts_block}

Return structured output only. Include results from ALL abstracts.
"""


def process_abstract_batch(args: tuple) -> list:
    """Process a batch of abstracts in one LLM call."""
    disease, batch = args  # batch = [{"pmid": ..., "abstract": ...}, ...]

    try:
        # Format all abstracts into one block
        abstracts_block = "\n\n".join([
            f"PMID: {item['pmid']}\nAbstract: {item['abstract']}"
            for item in batch
        ])

        prompt = BATCH_EXTRACTION_PROMPT_TEMPLATE.format(
            disease=disease,
            abstracts_block=abstracts_block
        )

        result = structured_llm.invoke(prompt)

        if result and result.root:
            extracted = []
            # Build a valid PMID set from this batch for validation
            valid_pmids = {item["pmid"] for item in batch}

            for entry in result.root:
                data = entry.model_dump(mode="json")
                # Only keep entries whose PMID is actually in this batch
                if data["pmid"] in valid_pmids:
                    extracted.append(data)
            return extracted

    except Exception as e:
        print(f"LLM batch extraction failed: {e}")
        return []