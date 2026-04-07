"""
╔══════════════════════════════════════════════════════════════╗
║         Drug Repurposing LangGraph Agent Pipeline            ║
║                                                              ║
║  Agents:                                                     ║
║   1. pubmed_node       → Fetch PubMed abstracts              ║
║   2. ner_node          → Extract disease-gene relationships  ║
║   3. dgidb_node        → Fetch drug-gene interactions        ║
║   4. kg_node           → Write everything to Neo4j           ║
║   5. reasoning_node    → GenAI repurposing candidates        ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
from langgraph.graph import StateGraph, END
from shared.config import warmup_models, free_reasoning_model
from shared.schemas import AgentState
from agents.pubmed_node    import pubmed_node
from agents.ner_node       import ner_node
from agents.dgidb_node     import dgidb_node
from agents.kg_node        import kg_node
from agents.reasoning_node import reasoning_node




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

    max_results = int(input("📄 How many abstracts to fetch? (default 300): ").strip() or "300")

    print(f"\n🚀 Starting Drug Repurposing Agent for: '{disease_input}'")
    print("="*60)

    warmup_models()
    free_reasoning_model()

    pipeline    = build_pipeline()
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
