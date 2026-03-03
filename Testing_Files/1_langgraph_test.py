from typing import TypedDict, List
from test_file import graph
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END




class DrugState(TypedDict):
    disease: str
    graph_candidates: List[dict]
    final_output: List[dict]


def graph_reasoning_node(state: DrugState):
    disease = state["disease"]

    results = graph.get_ranked_candidates(disease)

    return {
        "graph_candidates": results
    }

llm= ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

def explanation_node(state: DrugState):
    candidates = state["graph_candidates"][:5]

    prompt = f"""
    Based on the following drug repurposing candidates:

    {candidates}

    Explain biologically why these drugs may be good candidates.
    Keep explanation concise and scientific.
    """

    response = llm.invoke(prompt)

    return {
        "final_output": response.content
    }

g=StateGraph(DrugState)
g.add_node('graph_reasoning_node', graph_reasoning_node)
g.add_node('explanation_node', explanation_node)


g.add_edge(START, "graph_reasoning_node")
g.add_edge("graph_reasoning_node","explanation_node")
g.add_edge('explanation_node', END)

app=g.compile()

print(app)

initial_state = {
    "disease": "Alzheimer's disease",
    "graph_candidates": [],
    "final_output": []
}

result = app.invoke(initial_state)

print("\nFINAL OUTPUT:\n")
print(result["final_output"])
