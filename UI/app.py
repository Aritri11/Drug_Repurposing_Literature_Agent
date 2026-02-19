import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from workflow.graph import build_graph


st.title("💊 Drug Repurposing Agent")

disease = st.text_input("Enter disease")

if st.button("Run"):
    graph = build_graph()
    res = graph.invoke({"disease": disease})

    st.write("### Suggested Drugs")
    st.markdown(res["result"])

