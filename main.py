from workflow.graph import build_graph

app = build_graph()

out = app.invoke({"disease":"Alzheimer's disease"})
print(out["result"])
