#reasoning_node.py
from shared.schemas import AgentState
from shared.config import get_neo4j_driver, llm_reasoning
from shared.helpers import get_evidence_strength


# ======================================================
# 🤖 Reasoning Node
# Queries Neo4j for evidence paths and proposes
# top 5 repurposing candidates using LLM reasoning
#
# KEY DESIGN PRINCIPLE:
#   All ranking, deduplication, and score assignment is done
#   in Python — NEVER delegated to the LLM. The LLM only
#   writes narrative for the pre-ranked, pre-deduplicated top 5.
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

    WITH dr, t, g, ae, d,
        CASE
            WHEN (ae.direction = "UP"   AND toLower(t.interaction_type) CONTAINS "inhibit")    THEN 1.0
            WHEN (ae.direction = "UP"   AND toLower(t.interaction_type) CONTAINS "antagonist") THEN 1.0
            WHEN (ae.direction = "DOWN" AND toLower(t.interaction_type) CONTAINS "activate")   THEN 1.0
            WHEN (ae.direction = "DOWN" AND toLower(t.interaction_type) CONTAINS "agonist")    THEN 1.0
            WHEN (t.interaction_type = "" OR t.interaction_type IS NULL)                       THEN 0.0
            ELSE 0.2
        END AS direction_match_score,

        CASE
            WHEN t.interaction_type IS NULL THEN 0.0
            WHEN t.interaction_type = ""    THEN 0.0
            ELSE 1.0
        END AS interaction_known_score,

        CASE
            WHEN ae.evidence_count >= 10 THEN 1.0
            ELSE ae.evidence_count / 10.0
        END AS evidence_score,

        CASE
            WHEN t.interaction_score IS NULL THEN 0.0
            WHEN t.interaction_score > 1.0   THEN 1.0
            ELSE t.interaction_score
        END AS dgidb_score

    WITH dr, t, g, ae,
        direction_match_score,
        evidence_score,
        interaction_known_score,
        dgidb_score,
        round(
            direction_match_score   * 35
          + evidence_score          * 30
          + interaction_known_score * 20
          + dgidb_score             * 15
        ) AS computed_score

    WHERE computed_score > 20

    RETURN dr.name                  AS drug,
           g.symbol                 AS gene,
           t.interaction_type       AS drug_gene_interaction,
           t.approved               AS drug_approved,
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

        driver.close()

        # ── Step 1: Remove rows with empty/unknown interaction type ────────────
        rows = [
            row for row in rows
            if row.get("drug_gene_interaction") and row["drug_gene_interaction"].strip() != ""
        ]

        # ── Step 2: Remove direction mismatches (direction_match_score == 0) ───
        rows = [row for row in rows if row.get("direction_match_score", 0) > 0]

        # ── Step 3: Deduplicate by drug name — keep highest computed_score ─────
        # This is done in Python, NOT delegated to the LLM
        seen_drugs: dict[str, dict] = {}
        for row in rows:
            drug = row["drug"]
            if drug not in seen_drugs or row["computed_score"] > seen_drugs[drug]["computed_score"]:
                seen_drugs[drug] = row

        # ── Step 4: Sort and take top 5 — final, authoritative ranking ─────────
        top5 = sorted(seen_drugs.values(), key=lambda x: x["computed_score"], reverse=True)[:5]

        if not top5:
            report = (
                f"⚠️ No valid drug-gene-disease paths found in the knowledge graph for '{disease}'.\n"
                f"This means either DGIdb returned no interactions for the extracted genes,\n"
                f"or all candidates had unknown interaction types or mismatched directions.\n"
                f"Please fix the DGIdb connection and re-run."
            )
            print(report)
            return {**state, "final_report": report}

        # ── Step 5: Build structured evidence block for the LLM ───────────────
        # Scores, ranks, and labels are ALL pre-computed here.
        # The LLM receives fixed facts — it must only write narrative text.
        evidence_blocks = []
        for rank, row in enumerate(top5, start=1):
            drug_action    = row["drug_gene_interaction"]
            gene_direction = row["gene_disease_direction"]
            score          = int(row["computed_score"])
            d_score        = row["direction_match_score"]

            if d_score == 1.0:
                direction_label = (
                    f"✅ PERFECT MATCH — Gene is {gene_direction} in {disease}, "
                    f"Drug is a {drug_action} → correct therapeutic direction"
                )
            else:
                direction_label = (
                    f"⚠️ PARTIAL MATCH — Gene is {gene_direction} in {disease}, "
                    f"Drug action '{drug_action}' direction is partially unclear"
                )

            approved_label    = "✅ FDA Approved" if row.get("drug_approved") else "🔬 Experimental"
            evidence_strength = get_evidence_strength(score)
            gene_dir_label    = "Upregulated" if gene_direction == "UP" else "Downregulated"
            # Clean PMIDs — strip URLs, whitespace, and JSON garbage
            # that can accumulate if the KG stored dirty data
            raw_pmids = row.get("pmids") or []
            clean_pmids = []
            for p in raw_pmids:
                p = str(p).strip()
                # Extract bare numeric PMID from URLs like https://...pubmed/12345678
                if "pubmed" in p.lower():
                    p = p.rstrip("/").split("/")[-1]
                # Keep only if it looks like a real PMID (pure digits, 7-8 chars)
                if p.isdigit() and 6 <= len(p) <= 9:
                    clean_pmids.append(p)
            pmids_str = ", ".join(clean_pmids) if clean_pmids else "N/A"
            conflict_label    = " ⚠️ CONFLICTED DIRECTION" if row.get("conflicted") else ""

            block = f"""RANK {rank}:
Drug: {row['drug']}
Approval: {approved_label}
Gene: {row['gene']}{conflict_label}
Gene in {disease}: {gene_dir_label}
Drug→Gene interaction: {drug_action}
Direction match: {direction_label}
Evidence count: {row['evidence_count']}
Pre-computed score: {score}/100
Evidence strength: {evidence_strength}
PMIDs: {pmids_str}"""

            evidence_blocks.append(block)

            # Print the pre-computed top 5 to console for verification
            print(f"  #{rank} {row['drug']} → {row['gene']} | score={score} | {drug_action} | {gene_dir_label}")

        evidence_text = "\n\n".join(evidence_blocks)

        # ── Step 6: LLM prompt — narrative only, no ranking decisions ─────────
        # Format is enforced with an explicit TEMPLATE the LLM must fill in.
        # This prevents field collapsing by smaller models.
        reasoning_prompt = f"""You are a drug repurposing expert and clinical pharmacologist.

The top 5 repurposing candidates for {disease} have been pre-ranked by a validated algorithm.
Your ONLY job is to fill in the 4 narrative fields marked [FILL] for each rank.
All other fields are fixed — copy them VERBATIM. Do NOT change order, scores, or PMIDs.

Use EXACTLY this template for each rank, replacing [FILL] with your text:

---
RANK [N]
- Drug Name: [copy]
- Current Approved Use: [copy]
- Target Gene: [copy]
- Gene in {disease}: [copy]
- Drug→Gene Interaction: [copy]
- Direction Match: [copy]
- Pre-computed Score: [copy]
- Evidence Strength: [copy]
- Supporting PMIDs: [copy]
- Gene's Role in {disease}: [FILL — (a) normal function, (b) how dysregulation drives {disease}, (c) why valid target]
- Treatment Hypothesis: [FILL — 2-3 sentences: drug mechanism → gene correction → disease improvement]
- Known Risks in {disease}: [FILL — specific risks for this patient population, or "No known disease-specific risks identified"]
- Recommended Next Step: [FILL — specific and actionable: Phase II trial / animal model / IND application / in vitro validation]
---

PRE-RANKED EVIDENCE:
{evidence_text}

Now fill in the template for RANK 1 through RANK 5. Do not skip any field. Do not merge fields.
"""

        response      = llm_reasoning.invoke(reasoning_prompt)
        llm_content   = response.content

        # Strip <think> block if present (deepseek-r1 style)
        if "<think>" in llm_content:
            llm_content = llm_content.split("</think>")[-1].strip()

        print("\n" + "="*60)
        print(f"🏆 TOP 5 REPURPOSING CANDIDATES FOR: {disease}")
        print("="*60)
        print(llm_content)

        return {**state, "final_report": llm_content}

    except Exception as e:
        print(f"❌ Reasoning failed: {e}")
        driver.close()
        return {**state, "final_report": f"Reasoning failed: {e}"}
