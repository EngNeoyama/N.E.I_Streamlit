
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Caminho do índice
INDEX_PATH = "embeddings/faiss_index"

@st.cache_resource
def carregar_index():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Interface
st.set_page_config(page_title="Agente Técnico IA", layout="wide")
st.title("🔍 Agente de IA Técnico")

# Entrada da pergunta
query = st.text_input("Digite sua pergunta sobre os documentos carregados:")

# Processamento
if query:
    with st.spinner("Consultando..."):
        index = carregar_index()
        retriever = index.as_retriever()
        docs = retriever.get_relevant_documents(query)

        if docs:
            for i, doc in enumerate(docs):
                st.markdown(f"**Trecho {i+1}:**")
                st.write(doc.page_content)
                st.markdown("---")
        else:
            st.warning("Nenhum conteúdo relevante encontrado.")
