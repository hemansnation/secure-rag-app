import streamlit as st
from rag_pipeline import ingest_document, query_rag

st.title("Secure RAG Document Q&A System")

# Ingestion
uploaded_file = st.file_uploader("Upload PDF/Doc", type=["pdf", "docx"])
if uploaded_file and st.button("Ingest Document"):
    file_path = f"./data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    ingest_document(file_path)
    st.success("Document ingested! Chunks stored locally.")

# Query
query = st.text_input("Ask a question:")
if query and st.button("Query"):
    with st.spinner("Thinking..."):
        response = query_rag(query)
    st.markdown("**Answer:**")
    st.write(response)  # Citations appear inline, e.g., "...policy.[Chunk 1]"
    st.info("Citations come from retrieved chunks—verifiable and hallucination-reduced!")