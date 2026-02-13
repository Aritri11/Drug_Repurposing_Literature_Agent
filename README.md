# Drug Repurposing Agent

An agentic pipeline that uses large language models, knowledge graphs, and retrieval-augmented generation (RAG) to propose **drug repurposing candidates** for a given disease. The system retrieves biomedical literature and drug data, extracts drug–gene–disease relationships, builds a **Neo4j knowledge graph**, and runs an agentic reasoning workflow to rank promising drugs with explanations.[web:35][web:39][web:41]

---

##  Key Features

- **Multi-agent workflow** (retrieval, NER, relation extraction, reasoning) orchestrated via a graph-based controller.
- **Literature-aware**: pulls disease-specific abstracts from PubMed and structured data from drug/target resources.
- **Biomedical IE**: performs NER and relation extraction to obtain drug–gene–disease triplets from text.
- **Neo4j Knowledge Graph**: stores entities and relations for downstream querying and reasoning.
- **RAG + LLM reasoning**: uses vector search over graph/literature context plus an LLM to rank repurposing candidates.
- **Interactive UI**: Streamlit app to run the pipeline, inspect evidence, and view knowledge graph slices.

---

##  Project Structure

```text
drug-repurposing-agent/
│
├── agents/
│   ├── retriever.py           # Fetch PubMed data for a disease
│   ├── ner.py                 # Biomedical NER (drugs, genes, diseases)
│   ├── relation_extractor.py  # Extract drug–gene–disease relations
│   ├── reasoning.py           # LLM/RAG-based reasoning and ranking
│
├── kg/
│   ├── neo4j_client.py        # Thin wrapper around the Neo4j Python driver
│   └── kg_builder.py          # Build/update the Neo4j knowledge graph
│
├── rag/
│   └── vectorstore.py         # Vector store for documents / nodes (RAG)
│
├── workflow/
│   └── graph.py               # Orchestrates agents with a graph-style workflow
│
├── ui/
│   └── app.py                 # Streamlit UI for interactive use
│
├── config.py                  # Configuration (API keys, model names, Neo4j, etc.)
├── main.py                    # CLI entrypoint for running the pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
