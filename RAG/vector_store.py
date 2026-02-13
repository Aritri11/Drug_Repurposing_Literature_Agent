from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_vectorstore(docs):

    embed = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    return FAISS.from_texts(docs, embed)
