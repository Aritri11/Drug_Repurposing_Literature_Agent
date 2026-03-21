#kg_node.py
import re
import time
from shared.schemas import AgentState
from shared.config import get_neo4j_driver


# ======================================================
# 🟥 Knowledge Graph Node
# Writes Disease→Gene and Drug→Gene relationships to Neo4j
#
# Key improvements over original:
#   1. PMIDs are cleaned before storage — no URLs or garbage
#   2. Writes are batched (UNWIND) — one TX per entity type,
#      not one TX per row → much faster, less connection pressure
#   3. Retry logic on connection drop (OSError / ServiceUnavailable)
# ======================================================

# ── PMID cleaning ─────────────────────────────────────────────────────────────
def clean_pmid(raw: str) -> str | None:
    """
    Extract a bare numeric PMID from any string.
    Handles:
      - Plain PMIDs:          "41654938"         → "41654938"
      - PubMed URLs:          "https://.../41654938" → "41654938"
      - Prefixed strings:     ">41654938"        → "41654938"
      - Garbage like "}] [{": returns None
    """
    raw = str(raw).strip()
    # Extract last sequence of digits (handles URLs, prefixes, suffixes)
    digits = re.findall(r"\d+", raw)
    if not digits:
        return None
    candidate = digits[-1]  # Last digit group — typically the PMID in a URL
    # Valid PMIDs are 6–9 digits
    if 6 <= len(candidate) <= 9:
        return candidate
    return None


def clean_pmids(raw_list: list) -> list[str]:
    """Clean a list of raw PMID values, returning only valid ones."""
    seen = set()
    result = []
    for raw in (raw_list or []):
        pmid = clean_pmid(raw)
        if pmid and pmid not in seen:
            seen.add(pmid)
            result.append(pmid)
    return result


# ── Neo4j write with retry ─────────────────────────────────────────────────────
def run_with_retry(session, query, params=None, retries=3, delay=5.0):
    """Run a Cypher query with retry on transient connection errors."""
    for attempt in range(1, retries + 1):
        try:
            return session.run(query, **(params or {}))
        except Exception as e:
            is_connection = any(kw in str(e) for kw in ["OSError", "defunct", "ServiceUnavailable", "No data"])
            if is_connection and attempt < retries:
                print(f"  ⚠️ Neo4j connection error (attempt {attempt}/{retries}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


# ======================================================
# 🟥 KG Node
# ======================================================

def kg_node(state: AgentState) -> AgentState:

    disease       = state["disease"]
    disease_genes = state.get("disease_genes", [])
    drug_cands    = state.get("drug_candidates", [])

    print(f"\n🗄️  [KG Node] Writing to Neo4j...")

    driver = get_neo4j_driver()

    try:
        with driver.session() as session:

            # ── Constraints ───────────────────────────────────────────────────
            run_with_retry(session, "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE")
            run_with_retry(session, "CREATE CONSTRAINT gene_symbol  IF NOT EXISTS FOR (g:Gene)    REQUIRE g.symbol IS UNIQUE")
            run_with_retry(session, "CREATE CONSTRAINT drug_name    IF NOT EXISTS FOR (d:Drug)    REQUIRE d.name IS UNIQUE")

            # ── Step 1: Delete stale Disease→Gene edges ───────────────────────
            run_with_retry(session,
                "MATCH (d:Disease {name:$disease})-[r:ALTERS_EXPRESSION]->() DELETE r",
                {"disease": disease}
            )

            # ── Step 2: Delete stale Drug→Gene edges for this disease ─────────
            run_with_retry(session, """
                MATCH (d:Disease {name:$disease})-[:ALTERS_EXPRESSION]->(g:Gene)<-[r:TARGETS]-()
                DELETE r
            """, {"disease": disease})

            # ── Step 3: Build clean Disease→Gene payload ──────────────────────
            # Clean PMIDs here at write time so Neo4j never stores garbage
            dg_rows = []
            skipped = 0
            for item in disease_genes:
                pmid = clean_pmid(str(item.get("pmid", "")))
                if not pmid:
                    skipped += 1
                    continue
                dg_rows.append({
                    "gene":      item["gene"],
                    "direction": item["direction"],
                    "pmid":      pmid,
                })

            if skipped:
                print(f"  ⚠️ Skipped {skipped} disease-gene entries with invalid PMIDs")

            # ── Step 4: Write Disease→Gene in one batched transaction ─────────
            if dg_rows:
                run_with_retry(session, """
                UNWIND $rows AS row
                MERGE (d:Disease {name: $disease})
                MERGE (g:Gene    {symbol: row.gene})
                MERGE (d)-[r:ALTERS_EXPRESSION]->(g)
                SET r.pmids = CASE
                                WHEN r.pmids IS NULL          THEN [row.pmid]
                                WHEN NOT row.pmid IN r.pmids  THEN r.pmids + row.pmid
                                ELSE r.pmids
                              END,
                    r.evidence_count = CASE
                                WHEN r.pmids IS NULL          THEN 1
                                WHEN NOT row.pmid IN r.pmids  THEN size(r.pmids) + 1
                                ELSE size(r.pmids)
                              END,
                    r.up_count   = CASE WHEN row.direction = "UP"   THEN coalesce(r.up_count,   0) + 1 ELSE coalesce(r.up_count,   0) END,
                    r.down_count = CASE WHEN row.direction = "DOWN" THEN coalesce(r.down_count, 0) + 1 ELSE coalesce(r.down_count, 0) END,
                    r.direction  = CASE
                                WHEN row.direction = "UP"   AND coalesce(r.up_count,   0) + 1 >= coalesce(r.down_count, 0)     THEN "UP"
                                WHEN row.direction = "DOWN" AND coalesce(r.down_count, 0) + 1 >  coalesce(r.up_count,   0)     THEN "DOWN"
                                ELSE coalesce(r.direction, row.direction)
                              END,
                    r.conflicted = (coalesce(r.up_count, 0) > 0 AND coalesce(r.down_count, 0) > 0)
                """, {"disease": disease, "rows": dg_rows})

            print(f"  ✅ Wrote {len(dg_rows)} Disease→Gene relationships")

            # ── Step 5: Build Drug→Gene payload ──────────────────────────────
            dr_rows = []
            for item in drug_cands:
                dr_rows.append({
                    "drug":              item["drug"],
                    "gene":              item["gene"],
                    "interaction":       ",".join(item.get("interaction_type") or []),
                    "interaction_score": float(item.get("interaction_score") or 0.0),
                    "approved":          bool(item.get("approved", False)),
                })

            # ── Step 6: Write Drug→Gene in one batched transaction ────────────
            if dr_rows:
                run_with_retry(session, """
                UNWIND $rows AS row
                MERGE (dr:Drug {name: row.drug})
                MERGE (g:Gene  {symbol: row.gene})
                MERGE (dr)-[r:TARGETS]->(g)
                SET r.interaction_type  = row.interaction,
                    r.interaction_score = row.interaction_score,
                    r.approved          = row.approved
                """, {"rows": dr_rows})

            print(f"  ✅ Wrote {len(dr_rows)} Drug→Gene relationships")

        return {**state, "kg_status": "done"}

    except Exception as e:
        print(f"❌ Neo4j write failed: {e}")
        return {**state, "kg_status": "error"}

    finally:
        driver.close()
