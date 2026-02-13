from langgraph.graph import StateGraph

from agents.retriever import retrieve
from agents.ner import extract_entities
from agents.relation_extractor import extract_relations
from knowledge_graphs.kg_builder import build_kg
from agents.reasoning import reason

def build_graph():

    g = StateGraph(dict)

    g.add_node("retrieve", retrieve)
    g.add_node("ner", extract_entities)
    g.add_node("rel", extract_relations)
    g.add_node("kg", build_kg)
    g.add_node("reason", reason)

    g.set_entry_point("retrieve")

    g.add_edge("retrieve","ner")
    g.add_edge("ner","rel")
    g.add_edge("rel","kg")
    g.add_edge("kg","reason")

    return g.compile()
