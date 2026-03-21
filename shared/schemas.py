#schemas.py
from typing import List
from enum import Enum
from pydantic import BaseModel, RootModel
from typing_extensions import TypedDict
from shared.config import llm_ner, llm_reasoning



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


structured_llm = llm_ner.with_structured_output(GeneDirectionList)


# ======================================================
# 🗂️ LangGraph Agent State
# ======================================================

class AgentState(TypedDict):
    disease:          str          # Input disease name
    max_results:      int          # How many PubMed abstracts to fetch
    abstracts:        List[dict]   # [{pmid, abstract}, ...]
    disease_genes:    List[dict]   # [{gene, direction, evidence_type, pmid}, ...]
    drug_candidates:  List[dict]   # [{drug, gene, gene_direction, interaction_type}, ...]
    kg_status:        str          # "pending" | "done" | "error"
    final_report:     str          # Final repurposing output
