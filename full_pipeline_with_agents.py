"""
╔══════════════════════════════════════════════════════════════╗
║         Drug Repurposing LangGraph Agent Pipeline            ║
║                                                              ║
║  Nodes:                                                      ║
║   1. pubmed_node       → Fetch PubMed abstracts              ║
║   2. ner_node          → Extract disease-gene relationships  ║
║   3. dgidb_node        → Fetch drug-gene interactions        ║
║   4. kg_node           → Write everything to Neo4j           ║
║   5. reasoning_node    → GenAI repurposing candidates        ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from typing import List, Optional
from enum import Enum

from Bio import Entrez
import mygene

from pydantic import BaseModel, RootModel
from langchain_ollama import ChatOllama
from tqdm import tqdm

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from neo4j import GraphDatabase


# ======================================================
# 🔐 Environment Setup
# ======================================================

load_dotenv()

Entrez.email   = os.getenv("NCBI_EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j+s://your-instance.databases.neo4j.io")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

if not Entrez.email:
    raise ValueError("NCBI_EMAIL not found in .env file")


# ======================================================
# 🧠 LLM Setup
# ======================================================

llm = ChatOllama(model="llama3.1:8b", temperature=0)


# ======================================================
# 📦 Pydantic Schemas
# ======================================================

class Direction(str, Enum):
    UP   = "UP"
    DOWN = "DOWN"

class EvidenceType(str, Enum):
    BASELINE     = "BASELINE"
    INTERVENTION = "INTERVENTION"

class GeneDirection(BaseModel):
    gene:          str
    direction:     Direction
    evidence_type: EvidenceType
    pmid:          str

class GeneDirectionList(RootModel[List[GeneDirection]]):
    pass

structured_llm = llm.with_structured_output(GeneDirectionList)


# ======================================================
# 🗂️ LangGraph Agent State
# ======================================================

class AgentState(TypedDict):
    disease:          str                  # Input disease name
    max_results:      int                  # How many PubMed abstracts to fetch
    abstracts:        List[dict]           # [{pmid, abstract}, ...]
    disease_genes:    List[dict]           # [{gene, direction, evidence_type, pmid}, ...]
    drug_candidates:  List[dict]           # [{drug, gene, gene_direction, interaction_type}, ...]
    kg_status:        str                  # "pending" | "done" | "error"
    final_report:     str                  # Final repurposing output


# ======================================================
# 🔧 Helper: Neo4j Driver
# ======================================================

def get_neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )


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

    # DGIdb now uses GraphQL endpoint
    url = "https://dgidb.org/api/graphql"

    query = """
    {
      genes(names: %s) {
        nodes {
          name
          interactions {
            drug {
              name
            }
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
# 🔧 Helper: Process single abstract through LLM
# ======================================================

EXTRACTION_PROMPT_TEMPLATE = """
From the following PubMed abstract about {disease},

Identify genes whose expression levels are altered in patients with {disease}
under baseline disease conditions (NOT after drug treatment).

STRICT RULES:
- Extract only genes dysregulated in disease patients compared to controls.
- Ignore genes altered due to drug treatment or experimental manipulation.
- If gene is upregulated/activated/positively regulated (turning on gene expression) in disease → direction = "UP"
- If gene is downregulated/suppressed/negatively regulated (turning off gene expression) in disease → direction = "DOWN"
- evidence_type = "BASELINE" if naturally dysregulated in disease
- evidence_type = "INTERVENTION" if altered due to treatment or compound
- For pmid field, use exactly: {pmid}

Return structured output only.

Abstract:
{abstract}
"""

def process_single_abstract(args: tuple) -> list:
    """Process one abstract. Returns list of extracted gene dicts."""
    disease, pmid, abstract = args
    try:
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            disease=disease,
            pmid=pmid,
            abstract=abstract
        )
        result = structured_llm.invoke(prompt)
        if result and result.root:
            extracted = []
            for entry in result.root:
                data         = entry.model_dump(mode="json")
                data["pmid"] = pmid  # Always overwrite with ground truth
                extracted.append(data)
            return extracted
    except Exception as e:
        print(f"LLM extraction failed for PMID {pmid}: {e}")
    return []


# ======================================================
# 🟦 NODE 1: PubMed Node
# Fetches abstracts from PubMed for the given disease
# ======================================================

def pubmed_node(state: AgentState) -> AgentState:

    disease     = state["disease"]
    max_results = state.get("max_results", 20)

    print(f"\n📚 [PubMed Node] Fetching abstracts for: '{disease}'")

    search_term = f"""
    {disease}[Title/Abstract]
    AND ("expression level" OR upregulated OR downregulated)
    """

    try:
        search_handle  = Entrez.esearch(db="pubmed", term=search_term, retmax=max_results)
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results["IdList"]

        if not id_list:
            print("⚠️ No PubMed results found.")
            return {**state, "abstracts": []}

        fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml", retmode="xml")
        records      = Entrez.read(fetch_handle)
        fetch_handle.close()

        abstract_list = []
        for article in records["PubmedArticle"]:
            try:
                pmid          = str(article["MedlineCitation"]["PMID"])
                abstract_text = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
                full_abstract = " ".join([str(t) for t in abstract_text])
                abstract_list.append({"pmid": pmid, "abstract": full_abstract})
            except KeyError:
                continue

        print(f"✅ Fetched {len(abstract_list)} abstracts.")
        return {**state, "abstracts": abstract_list}

    except Exception as e:
        print(f"❌ PubMed fetch failed: {e}")
        return {**state, "abstracts": []}


# ======================================================
# 🟩 NODE 2: NER Node
# Extracts disease-gene relationships from abstracts
# Uses parallel processing for speed
# ======================================================

def ner_node(state: AgentState) -> AgentState:

    disease       = state["disease"]
    abstract_list = state.get("abstracts", [])

    if not abstract_list:
        print("⚠️ [NER Node] No abstracts to process.")
        return {**state, "disease_genes": []}

    print(f"\n🧬 [NER Node] Extracting genes from {len(abstract_list)} abstracts (parallel)...")

    # Build args list for parallel processing
    args_list = [
        (disease, item["pmid"], item["abstract"])
        for item in abstract_list
    ]

    all_extracted = []

    # Parallel LLM calls — 5 workers keeps it fast without overloading Ollama
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_abstract, args): args for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing abstracts"):
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


# ======================================================
# 🟨 NODE 3: DGIdb Node
# Fetches drug-gene interactions for all extracted genes
# Uses a single batch API call instead of one per gene
# ======================================================

def dgidb_node(state: AgentState) -> AgentState:

    disease_genes = state.get("disease_genes", [])

    if not disease_genes:
        print("⚠️ [DGIdb Node] No genes to query.")
        return {**state, "drug_candidates": []}

    # Get unique gene list
    gene_list = list(set(item["gene"] for item in disease_genes))

    print(f"\n💊 [DGIdb Node] Querying DGIdb for {len(gene_list)} genes in one batch call...")

    # ONE batch API call for all genes
    interactions_map = fetch_dgidb_interactions_batch(gene_list)

    # Build gene direction lookup
    gene_direction_map = {item["gene"]: item["direction"] for item in disease_genes}

    drug_candidates = []

    for gene, interactions in interactions_map.items():
        gene_direction = gene_direction_map.get(gene, "UNKNOWN")

        for interaction in interactions:
            drug  = interaction.get("drug", {}).get("name", "")
            types = [t["type"] for t in interaction.get("interactionTypes", [])]

            if drug:
                drug_candidates.append({
                    "drug":             drug,
                    "gene":             gene,
                    "gene_direction":   gene_direction,
                    "interaction_type": types,
                    "interaction_score": interaction.get("interactionScore", 0.0)  # ← add this
                })

    print(f"✅ Found {len(drug_candidates)} drug-gene candidates.")
    return {**state, "drug_candidates": drug_candidates}


# ======================================================
# 🟥 NODE 4: Knowledge Graph Node
# Writes Disease→Gene and Drug→Gene relationships to Neo4j
# ======================================================

def kg_node(state: AgentState) -> AgentState:

    disease       = state["disease"]
    disease_genes = state.get("disease_genes", [])
    drug_cands    = state.get("drug_candidates", [])

    print(f"\n🗄️  [KG Node] Writing to Neo4j...")

    driver = get_neo4j_driver()

    try:
        with driver.session() as session:

            # --- Constraints ---
            session.run("CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE")
            session.run("CREATE CONSTRAINT gene_symbol IF NOT EXISTS FOR (g:Gene) REQUIRE g.symbol IS UNIQUE")
            session.run("CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE")

            # --- Clear old Disease→Gene relationships ---
            session.run(
                "MATCH (d:Disease {name:$disease})-[r:ALTERS_EXPRESSION]->() DELETE r",
                disease=disease
            )

            # --- Insert Disease→Gene relationships ---
            disease_gene_query = """
            MERGE (d:Disease {name:$disease})
            MERGE (g:Gene {symbol:$gene})
            MERGE (d)-[r:ALTERS_EXPRESSION]->(g)
            SET r.direction = $direction,
                r.pmids = CASE
                            WHEN r.pmids IS NULL THEN [$pmid]
                            WHEN NOT $pmid IN r.pmids THEN r.pmids + $pmid
                            ELSE r.pmids
                          END,
                r.evidence_count = CASE
                            WHEN r.pmids IS NULL THEN 1
                            WHEN NOT $pmid IN r.pmids THEN size(r.pmids) + 1
                            ELSE size(r.pmids)
                          END,
                r.confidence_score = CASE
                            WHEN r.pmids IS NULL THEN 0.1
                            WHEN NOT $pmid IN r.pmids THEN (size(r.pmids) + 1) * 0.1
                            ELSE size(r.pmids) * 0.1
                          END
            """

            for item in disease_genes:
                session.run(
                    disease_gene_query,
                    disease=disease,
                    gene=item["gene"],
                    direction=item["direction"],
                    pmid=item["pmid"]
                )

            print(f"  ✅ Wrote {len(disease_genes)} Disease→Gene relationships")

            # --- Insert Drug→Gene relationships ---
            drug_gene_query = """
            MERGE (dr:Drug {name:$drug})
            MERGE (g:Gene {symbol:$gene})
            MERGE (dr)-[r:TARGETS]->(g)
            SET r.interaction_type = $interaction,
                r.interaction_score = $interaction_score
            """

            for item in drug_cands:
                session.run(
                    drug_gene_query,
                    drug=item["drug"],
                    gene=item["gene"],
                    interaction=",".join(item["interaction_type"]),
                    interaction_score=item.get("interaction_score", 0.0)
                )

            print(f"  ✅ Wrote {len(drug_cands)} Drug→Gene relationships")

        return {**state, "kg_status": "done"}

    except Exception as e:
        print(f"❌ Neo4j write failed: {e}")
        return {**state, "kg_status": "error"}

    finally:
        driver.close()


# ======================================================
# 🤖 NODE 5: Reasoning Node
# Queries Neo4j for evidence paths and proposes
# top 5 repurposing candidates using LLM reasoning
# ======================================================

def reasoning_node(state: AgentState) -> AgentState:

    disease   = state["disease"]
    kg_status = state.get("kg_status", "pending")

    if kg_status != "done":
        print("⚠️ [Reasoning Node] KG not ready, skipping reasoning.")
        return {**state, "final_report": "KG population failed. Cannot reason."}

    print(f"\n🤖 [Reasoning Node] Querying graph and reasoning over evidence...")

    driver = get_neo4j_driver()

    try:
        with driver.session() as session:
            result = session.run("""
               MATCH (dr:Drug)-[t:TARGETS]->(g:Gene)<-[ae:ALTERS_EXPRESSION]-(d:Disease {name: $disease})

    // ── Factor 1: Direction match ──────────────────────────────
    // INHIBITOR drug + UP gene   = match ✅
    // ACTIVATOR drug + DOWN gene = match ✅
    WITH dr, t, g, ae, d,
        CASE
            WHEN (ae.direction = "UP"   AND t.interaction_type CONTAINS "inhibit")  THEN 1.0
            WHEN (ae.direction = "UP"   AND t.interaction_type CONTAINS "antagoni") THEN 1.0
            WHEN (ae.direction = "DOWN" AND t.interaction_type CONTAINS "activat")  THEN 1.0
            WHEN (ae.direction = "DOWN" AND t.interaction_type CONTAINS "agonist")  THEN 1.0
            WHEN (t.interaction_type = "" OR t.interaction_type IS NULL)            THEN 0.0
            ELSE 0.2   // known but unclear direction
        END AS direction_match_score,

        // ── Factor 2: Interaction type is known ────────────────
        CASE
            WHEN t.interaction_type IS NULL THEN 0.0
            WHEN t.interaction_type = ""    THEN 0.0
            ELSE 1.0
        END AS interaction_known_score,

        // ── Factor 3: Evidence count (capped at 10 papers) ─────
        CASE
            WHEN ae.evidence_count >= 10 THEN 1.0
            ELSE ae.evidence_count / 10.0
        END AS evidence_score,
        
        // ── Factor 4: DGIdb interaction score (normalized 0-1) ──
        CASE
            WHEN t.interaction_score IS NULL THEN 0.0
            WHEN t.interaction_score > 1.0   THEN 1.0
            ELSE t.interaction_score
        END AS dgidb_score

    // ── Combine into final pre-computed score (0–100) ──────────
    WITH dr, t, g, ae,
        round(
            ( direction_match_score   * 35   // 35% — direction logic
            + evidence_score          * 30   // 30% — literature support  
            + interaction_known_score * 20   // 20% — data quality
            + dgidb_score             * 15   // 15% — DGIdb confidence
            )
        ) AS computed_score,
        direction_match_score,
        evidence_score,
        interaction_known_score

    // Only return candidates with a meaningful score
    WHERE computed_score > 20

    RETURN dr.name                  AS drug,
           g.symbol                 AS gene,
           t.interaction_type       AS drug_gene_interaction,
           ae.direction             AS gene_disease_direction,
           ae.evidence_count        AS evidence_count,
           ae.pmids                 AS pmids,
           direction_match_score    AS direction_match_score,
           evidence_score           AS evidence_score,
           interaction_known_score  AS interaction_known_score,
           computed_score           AS computed_score

    ORDER BY computed_score DESC
    LIMIT 100
""", disease=disease)
            rows = result.data()
            # Remove rows with empty or unknown interaction type
            rows = [
                row for row in rows
                if row.get("drug_gene_interaction") and row["drug_gene_interaction"].strip() != ""
            ]

        driver.close()

        # ✅ HARD STOP — only if truly empty
        if not rows:
            report = (
                f"⚠️ No drug-gene-disease paths found in the knowledge graph for '{disease}'.\n"
                f"This means DGIdb returned no interactions for the extracted genes.\n"
                f"Cannot generate repurposing candidates without graph evidence.\n"
                f"Please fix the DGIdb connection and re-run."
            )
            print(report)
            return {**state, "final_report": report}

        # ✅ Only reaches here if real graph evidence exists
        evidence_lines = []
        for row in rows:
            # Describe direction match clearly for LLM
            if row["direction_match_score"] == 1.0:
                direction_label = "✅ PERFECT MATCH"
            elif row["direction_match_score"] == 0.2:
                direction_label = "⚠️ PARTIAL MATCH (interaction direction unclear)"
            else:
                direction_label = "❌ NO MATCH"
            line = (
                f"Drug: {row['drug']} | "
                f"Gene: {row['gene']} | "
                f"Drug→Gene: {row['drug_gene_interaction'] or 'UNKNOWN'} | "
                f"Gene in {disease}: {row['gene_disease_direction']} | "
                f"Direction match: {direction_label} | "
                f"Evidence count: {row['evidence_count']} | "
                f"Pre-computed score: {row['computed_score']}/100 | "
                f"PMIDs: {', '.join(row['pmids'] or [])}"
            )
            evidence_lines.append(line)

        evidence_text = "\n".join(evidence_lines)

        reasoning_prompt = f"""
You are a drug repurposing expert.

Below is knowledge graph evidence for {disease}.
Each line shows: Drug → Gene Target → Gene's role in {disease}
The evidence below has already been scored by a multi-factor algorithm:
  - Direction match (40%): Does drug action oppose gene's disease role?
  - Literature support (35%): How many PubMed papers confirm gene-disease link?
  - Data quality (25%): Is the drug-gene interaction type explicitly known?

CRITICAL RULE: 
- If Drug→Gene interaction type is empty, unknown, or unspecified → SKIP that candidate entirely.
- Only reason over candidates where the interaction type is explicitly known (inhibitor, activator, agonist, antagonist etc.)
- Do NOT use your own training knowledge to fill in missing interaction types.
- Use the pre-computed score as your primary ranking basis
- Do NOT invent or modify scores
- Do NOT use candidates where Direction match = ❌ NO MATCH
- Do NOT fill in missing interaction types from your own knowledge
- Every claim must trace back to the PMIDs listed


KEY LOGIC:
- If gene is DOWNREGULATED in disease + Drug ACTIVATES gene → ✅ Good candidate
- If gene is UPREGULATED in disease   + Drug INHIBITS gene  → ✅ Good candidate
- Empty interaction type → ❌ Skip entirely
- Mismatched direction = ❌ Wrong candidate

EVIDENCE (sorted by pre-computed score, highest first):
{evidence_text}

Select the TOP 5 candidates with the highest pre-computed scores and ✅ PERFECT MATCH direction.

RANK X:
- Drug Name: [just the drug name alone]
- Current Approved Use: [approved indication or "Not yet approved / Experimental"]
- Target Gene: [just the gene symbol]
- Gene's Role in {disease}: [explanation]
- Direction Match: [explanation]
- Pre-computed Score: [copy exactly from evidence]
- Evidence Strength: High / Medium / Low
- Supporting PMIDs: [copy exactly]
- Next Step: [recommendation]

Format clearly as RANK 1 through RANK 5.
"""

        response     = llm.invoke(reasoning_prompt)
        final_report = response.content

        print("\n" + "="*60)
        print(f"🏆 TOP 5 REPURPOSING CANDIDATES FOR: {disease}")
        print("="*60)
        print(final_report)

        return {**state, "final_report": final_report}

    except Exception as e:
        print(f"❌ Reasoning failed: {e}")
        driver.close()
        return {**state, "final_report": f"Reasoning failed: {e}"}

# ======================================================
# 🔗 Build the LangGraph Pipeline
# ======================================================

def build_pipeline() -> StateGraph:

    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("pubmed",    pubmed_node)
    workflow.add_node("ner",       ner_node)
    workflow.add_node("dgidb",     dgidb_node)
    workflow.add_node("kg",        kg_node)
    workflow.add_node("reasoning", reasoning_node)

    # Wire the edges — sequential pipeline
    workflow.set_entry_point("pubmed")
    workflow.add_edge("pubmed",    "ner")
    workflow.add_edge("ner",       "dgidb")
    workflow.add_edge("dgidb",     "kg")
    workflow.add_edge("kg",        "reasoning")
    workflow.add_edge("reasoning", END)

    return workflow.compile()


# ======================================================
# 🚀 Main Entry Point
# ======================================================

if __name__ == "__main__":

    # Accept disease from command line or prompt
    if len(sys.argv) > 1:
        disease_input = " ".join(sys.argv[1:])
    else:
        disease_input = input("🔬 Enter disease name: ").strip()

    if not disease_input:
        print("❌ No disease name provided.")
        sys.exit(1)

    max_results = int(input("📄 How many abstracts to fetch? (default 20): ").strip() or "20")

    print(f"\n🚀 Starting Drug Repurposing Agent for: '{disease_input}'")
    print("="*60)

    # Build and run the pipeline
    pipeline = build_pipeline()

    final_state = pipeline.invoke({
        "disease":         disease_input,
        "max_results":     max_results,
        "abstracts":       [],
        "disease_genes":   [],
        "drug_candidates": [],
        "kg_status":       "pending",
        "final_report":    ""
    })

    print("\n✅ Pipeline complete.")
    print(f"📊 Abstracts fetched:       {len(final_state['abstracts'])}")
    print(f"🧬 Disease-gene pairs:      {len(final_state['disease_genes'])}")
    print(f"💊 Drug-gene candidates:    {len(final_state['drug_candidates'])}")
    print(f"🗄️  KG status:               {final_state['kg_status']}")