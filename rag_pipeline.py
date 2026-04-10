import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# Load env
load_dotenv()

# ✅ Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ✅ Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------------------------------
# Load documents
# -------------------------------
def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source_file"] = file

            docs.extend(loaded_docs)
    return docs

# -------------------------------
# Split documents
# -------------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

# -------------------------------
# Create vector store
# -------------------------------
def create_vector_store(chunks):
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    return vectorstore

# -------------------------------
# Retrieve documents
# -------------------------------
def retrieve(query, vectorstore, k=3):
    return vectorstore.similarity_search(query, k=k)

# -------------------------------
# Format sources
# -------------------------------
def format_sources(docs):
    sources = []
    for doc in docs:
        file = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "N/A")
        sources.append(f"{file} (Page {page})")
    
    return list(set(sources))

# -------------------------------
# Generate answer (Groq)
# -------------------------------
def generate_answer(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = format_sources(docs)

    prompt = f"""
You are an enterprise AI assistant.

STRICT RULES:
1. Answer ONLY from the given context
2. If answer is not in context, say "I don't know"
3. Do NOT hallucinate
4. Keep answer concise

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer = response.choices[0].message.content

    except Exception as e:
        answer = f"❌ ERROR: {str(e)}"

    answer += "\n\n📄 Sources:\n"
    for s in sources:
        answer += f"- {s}\n"

    return answer