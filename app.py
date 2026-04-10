import streamlit as st
from rag_pipeline import load_documents, split_documents, create_vector_store, retrieve, generate_answer

st.set_page_config(page_title="Enterprise Q&A System")

st.title("📚 Enterprise Knowledge Base Q&A")

# -------------------------------
# Setup
# -------------------------------
@st.cache_resource
def setup():
    docs = load_documents("data")
    chunks = split_documents(docs)
    vectorstore = create_vector_store(chunks)
    return vectorstore

vectorstore = setup()

# -------------------------------
# User Input
# -------------------------------
query = st.text_input("Ask your question:")

if query:
    docs = retrieve(query, vectorstore)
    answer = generate_answer(query, docs)

    st.subheader("Answer:")
    st.write(answer)

    with st.expander("📄 Source Details"):
        for i, doc in enumerate(docs):
            st.write(f"{i+1}. {doc.metadata}")