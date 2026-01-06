import os
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from pypdf import PdfReader

# Optional OpenAI
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except ImportError:
    OpenAIEmbeddings = None
    ChatOpenAI = None

# Local LLM
from transformers import pipeline


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG Document Assistant", layout="wide")
st.title("📄 RAG Document Assistant")
st.caption("Local / OpenAI embeddings • FAISS • Hallucination-safe RAG")

# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.header("⚙️ Configuration")

embedding_mode = st.sidebar.radio(
    "Embedding backend",
    ["Local (HuggingFace)", "OpenAI"]
)

llm_mode = st.sidebar.radio(
    "Answer generation",
    ["Local LLM", "OpenAI LLM"]
)

data_source = st.sidebar.radio(
    "Data source",
    ["Upload PDF", "Use text file"]
)

# -----------------------------
# Load Document Text
# -----------------------------
raw_text = ""

if data_source == "Upload PDF":
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf:
        reader = PdfReader(uploaded_pdf)
        for page in reader.pages:
            raw_text += page.extract_text() or ""

else:
    DATA_FILE = "rag_test_data.txt"
    if not os.path.exists(DATA_FILE):
        st.error("rag_test_data.txt not found")
        st.stop()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

if not raw_text.strip():
    st.info("📥 Please upload a PDF or select a text file")
    st.stop()

# -----------------------------
# Chunking
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)

chunks = splitter.split_text(raw_text)
documents = [Document(page_content=c) for c in chunks]

st.success(f"✅ Document split into {len(documents)} chunks")

# -----------------------------
# Embeddings
# -----------------------------
if embedding_mode == "Local (HuggingFace)":
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
else:
    if OpenAIEmbeddings is None or not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI embeddings selected but API key not available")
        st.stop()
    embeddings = OpenAIEmbeddings()

# -----------------------------
# Vector Store
# -----------------------------
vectorstore = FAISS.from_documents(documents, embeddings)

# -----------------------------
# Question
# -----------------------------
query = st.text_input("Ask a question")

if query:
    retrieved_docs = vectorstore.similarity_search(query, k=3)

    st.subheader("🔎 Retrieved Context")
    context = ""

    for i, doc in enumerate(retrieved_docs, 1):
        st.markdown(f"**Chunk {i}:**")
        st.write(doc.page_content)
        context += doc.page_content + "\n"

    # -----------------------------
    # Hallucination-Safe Prompt
    # -----------------------------
    prompt = f"""
You are a strict question-answering system.

RULES:
- Answer ONLY using the information provided in the context.
- If the answer is NOT present in the context, say exactly:
  "The document does not contain this information."
- Do NOT use prior knowledge.
- Do NOT guess.
- Do NOT add external facts.

Context:
{context}

Question:
{query}
"""

    # -----------------------------
    # Answer Generation
    # -----------------------------
    st.subheader("🧠 Final Answer")

    if llm_mode == "Local LLM":
        with st.spinner("Generating answer (local LLM)..."):
            generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                max_new_tokens=200
            )

            result = generator(prompt)[0]["generated_text"]
            st.write(result)

    else:
        if ChatOpenAI is None or not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI LLM selected but API key not available")
            st.stop()

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        answer = llm.invoke(prompt).content
        st.write(answer)
