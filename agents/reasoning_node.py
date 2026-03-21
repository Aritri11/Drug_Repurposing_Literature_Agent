from shared.schemas import AgentState
from shared.config import get_neo4j_driver, llm_reasoning
from shared.helpers import get_evidence_strength


# ======================================================
# 🤖 Reasoning Node
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

            # Remove rows with empty or unknown interaction type
            rows = [
                row for row in rows
                if row.get("drug_gene_interaction") and row["drug_gene_interaction"].strip() != ""
            ]

            # Deduplicate by drug+gene combination — keep highest score
            seen = {}
            for row in rows:
                key = f"{row['drug']}_{row['gene']}"
                if key not in seen or row["computed_score"] > seen[key]["computed_score"]:
                    seen[key] = row

            rows = list(seen.values())
            rows.sort(key=lambda x: x["computed_score"], reverse=True)

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
            drug_action    = row["drug_gene_interaction"]
            gene_direction = row["gene_disease_direction"]

            if row["direction_match_score"] == 1.0:
                direction_label = (
                    f"✅ PERFECT MATCH — "
                    f"Gene is {gene_direction} in {disease}, "
                    f"Drug is a {drug_action} → correct therapeutic direction"
                )
            elif row["direction_match_score"] == 0.2:
                direction_label = (
                    f"⚠️ PARTIAL MATCH — "
                    f"Gene is {gene_direction} in {disease}, "
                    f"Drug action '{drug_action}' direction is unclear"
                )
            else:
                direction_label = (
                    f"❌ NO MATCH — "
                    f"Gene is {gene_direction} in {disease}, "
                    f"Drug '{drug_action}' action works in wrong direction"
                )

            approved_label    = "✅ FDA Approved" if row.get("drug_approved") else "🔬 Experimental"
            evidence_strength = get_evidence_strength(row["computed_score"])
            conflicted        = row.get("conflicted", False)
            conflict_label    = " ⚠️ CONFLICTED DIRECTION" if conflicted else ""
            gene_dir_label = "Upregulated" if row["gene_disease_direction"] == "UP" else "Downregulated"

            line = (
                f"Drug: {row['drug']} | "
                f"Approval: {approved_label} | "
                f"Gene: {row['gene']}{conflict_label} | "
                f"Drug→Gene: {drug_action} | "
                f"Gene in {disease}: {gene_dir_label} | "
                f"Direction match: {direction_label} | "
                f"Evidence count: {row['evidence_count']} | "
                f"Pre-computed score: {row['computed_score']}/100 | "
                f"Evidence Strength: {evidence_strength} | "
                f"PMIDs: {', '.join(row['pmids'] or [])}"
            )
            evidence_lines.append(line)

        evidence_text = "\n".join(evidence_lines)

        reasoning_prompt = f"""
You are a drug repurposing expert and a clinical pharmacologist.

Below is knowledge graph evidence for {disease}.
Each line shows: Drug → Gene Target → Gene's role in {disease}
The evidence below or the pre-computed score (0-100) for each candidate has already been scored by a multi-factor algorithm:
  - Direction match (35%): Does drug action oppose gene's disease role?
  - Literature support (30%): How many PubMed papers confirm gene-disease link? (capped at 10)
  - Data quality (20%): Is the drug-gene interaction type explicitly known?
  - DGIdb confidence (15%): DGIdb's own internal interaction confidence score

CRITICAL RULE:
- If Drug→Gene interaction type is empty, unknown, or unspecified → SKIP that candidate entirely.
- Only reason over candidates where the interaction type is explicitly known (inhibitor, activator, agonist, antagonist etc.)
- Do NOT use your own training knowledge to fill in missing interaction types.
- Use the pre-computed score as your primary ranking basis
- Do NOT invent or modify scores
- Do NOT use candidates where Direction match = ❌ NO MATCH
- Do NOT fill in missing interaction types from your own knowledge
- Every claim must trace back to the PMIDs listed
- If a gene is marked ⚠️ CONFLICTED DIRECTION, lower its confidence —
  literature disagrees on whether it is UP or DOWN in this disease.
  Mention the conflict in your explanation.

KEY LOGIC:
- If gene is DOWNREGULATED in disease + Drug ACTIVATES gene → ✅ Good candidate
- If gene is UPREGULATED in disease   + Drug INHIBITS gene  → ✅ Good candidate
- Empty interaction type → ❌ Skip entirely
- Mismatched direction = ❌ Wrong candidate

EVIDENCE (sorted by pre-computed score, highest first):
{evidence_text}

IMPORTANT: Each drug should appear only ONCE in the top 5.
If the same drug targets multiple genes, pick the gene-disease path
with the highest pre-computed score and use only that one.

The evidence below is ALREADY sorted by pre-computed score, highest first.
You MUST present the top 5 in EXACTLY the order they appear in the evidence.
Do NOT re-rank, re-sort, or change the order for any reason.
RANK 1 = first item in evidence, RANK 2 = second item, and so on.

RANK X:
- Drug Name: [just the drug name alone]
- Current Approved Use: [copy exactly from evidence — either ✅ FDA Approved or 🔬 Experimental]
- Target Gene: [just the gene symbol]
- Gene's Role in {disease}:
  [Start with Upregulated/Downregulated as stated in evidence. Then explain:
   (a) what this gene normally does in healthy tissue
   (b) how its dysregulation contributes to {disease} pathology
   (c) why it is a valid therapeutic target]
- Treatment Hypothesis:
  [In 2-3 sentences, explain the precise mechanism by which this drug could 
   benefit a {disease} patient. Connect:
   drug mechanism → gene correction → disease pathology improvement.
   Example: "By inhibiting X, this drug reduces Y, which directly addresses 
   the Z dysfunction seen in {disease}."]
- Known Risks in {disease}:
  [List any known side effects of this drug that could specifically worsen 
   {disease} symptoms or create dangerous interactions in this patient 
   population. If none are known, state: "No known disease-specific risks 
   identified — general safety profile applies."]
- Direction Match: [copy exactly from evidence — includes gene direction and drug action]
- Pre-computed Score (Confidence Score[0-100]): [copy exactly from evidence]
- Evidence Strength: [copy exactly from evidence — 🟢 High / 🟡 Medium / 🔴 Low]
- Supporting PMIDs: [copy exactly]
- Recommended Next Step:
  [Give a SPECIFIC, actionable recommendation — not generic. Choose the most 
   appropriate one based on evidence strength and approval status:
   • If FDA Approved + High evidence → suggest specific Phase II/III trial design
   • If FDA Approved + Medium evidence → suggest in vivo animal model study
   • If Experimental + High evidence → suggest IND application pathway
   • If Experimental + Low evidence → suggest in vitro mechanistic validation
   Always mention the gene target and disease context specifically.]


Format clearly as RANK 1 through RANK 5.
"""

        response     = llm_reasoning.invoke(reasoning_prompt)
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
