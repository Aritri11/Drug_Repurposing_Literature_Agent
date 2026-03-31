# Drug Repurposing Agent

An agentic pipeline that uses large language models, knowledge graphs, and retrieval-augmented generation (RAG) to propose **drug repurposing candidates** for a given disease. The system retrieves biomedical literature and drug data, extracts drug–gene–disease relationships, builds a **Neo4j knowledge graph**, and runs an agentic reasoning workflow to rank promising drugs with explanations.

---

##  Key Features

- **Multi-agent workflow** (retrieval, NER, relation extraction, reasoning) orchestrated via a graph-based controller.
- **Literature-aware**: pulls disease-specific abstracts from PubMed and structured data from drug/target resources.
- **Biomedical IE**: performs NER and relation extraction to obtain drug–gene–disease triplets from text.
- **Neo4j Knowledge Graph**: stores entities and relations for downstream querying and reasoning.
- **LLM reasoning**: uses LLM to rank repurposing candidates.

---

##  Project Structure

```text
drug-repurposing-agent/
│
├── agents/
│   ├── config.py    # Configuration (API keys, model names, Neo4j, etc.)       
│   ├── helpers.py          
│   ├── schemas.py         
│   
├── agents/
│   ├── pubmed_node.py          # Fetch PubMed data for a disease
│   ├── ner.py                  # Biomedical NER (drugs, genes, diseases)
│   ├── kg_node.py              # Extract drug–gene–disease relations
│   ├── reasoning.py           # LLM/RAG-based reasoning and ranking
|   ├── dgidb_node.py           #DGIDB parser
│                
├── pipeline.py                # CLI entrypoint for running the pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
