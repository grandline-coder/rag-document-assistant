# 📄 RAG Document Assistant (Local & OpenAI)

## 🔍 Overview
RAG Document Assistant is a Retrieval-Augmented Generation (RAG) based application that allows users to upload PDFs or use text files, ask natural language questions, retrieve relevant context using vector search, and generate hallucination-safe answers.

The system supports both **local embeddings** (free, offline) and **OpenAI embeddings** (production-grade), making it suitable for real-world deployment and interviews.

---

## 🧠 What is RAG?
Retrieval-Augmented Generation (RAG) combines:
1. Information Retrieval (vector similarity search)
2. Text Generation (LLMs)

Instead of relying purely on an LLM’s internal knowledge, RAG retrieves relevant document chunks and injects them into the prompt, producing grounded and factual answers.

---

## 🏗️ System Architecture

PDF / Text  
→ Chunking  
→ Vector Embeddings (Local / OpenAI)  
→ FAISS Vector Database  
→ Top-K Retrieved Chunks  
→ LLM with Guardrails  
→ Final Answer / Refusal  

---

## ⚙️ Key Features

- 📄 PDF Upload & Text File Support  
- ✂️ Intelligent Chunking with Overlap  
- 🧮 Local (HuggingFace) & OpenAI Embeddings  
- ⚡ FAISS Vector Search  
- 🛡️ Strict Hallucination Guardrails  
- 🔁 Modular, Model-Agnostic Design  

---

## 🔁 Embedding Strategy

| Mode | Usage |
|---|---|
| Local (HuggingFace) | Development, testing, zero cost |
| OpenAI | Production, higher semantic accuracy |

Switching embedding backends requires no architectural change.

---

## 🛡️ Hallucination-Safe Answering

If the answer is not present in the retrieved context, the system responds with:

> “The document does not contain this information.”

This prevents post-retrieval hallucinations and improves trustworthiness.

---

## 📸 Screenshots (Visual Walkthrough)

> Add screenshots in a `screenshots/` folder and reference them below.  
> This section greatly improves recruiter and interviewer understanding.

### 1️⃣ Application Home & Configuration
![App Home](screenshots/01_app_home.png)

Shows:
- Embedding backend selection (Local / OpenAI)
- LLM selection
- Data source selection (PDF / Text)

---

### 2️⃣ PDF Upload & Chunk Processing
![PDF Upload](screenshots/02_pdf_upload.png)

Shows:
- PDF upload via UI
- Automatic text extraction
- Chunk count confirmation

---

### 3️⃣ Semantic Retrieval (FAISS)
![Retrieval](screenshots/03_retrieval.png)

Shows:
- Top-K retrieved chunks
- Semantic (meaning-based) search
- Grounded context display

---

### 4️⃣ Final Answer with Guardrails
![Final Answer](screenshots/04_final_answer.png)

Shows:
- Hallucination-safe answer
- Explicit refusal when answer is not in document

---

### 5️⃣ Negative Test (No Hallucination)
![Negative Test](screenshots/05_negative_test.png)

Shows:
- Question not present in document
- Correct refusal instead of hallucination

---

## 🧪 Evaluation Metrics (IMPORTANT)

This project is evaluated using **behavioral and retrieval-focused metrics**, not traditional accuracy alone.

### 1️⃣ Retrieval Quality
- Top-K relevance check
- Manual inspection of retrieved chunks
- Ensures semantic correctness

### 2️⃣ Grounded Answer Rate
- Percentage of answers strictly derived from retrieved context
- Target: High grounding, zero hallucination

### 3️⃣ Refusal Accuracy (Negative Testing)
- System correctly refuses when data is missing
- Example: “Who invented RAG?”

### 4️⃣ Latency
- Embedding time
- Retrieval response time
- Answer generation time

### 5️⃣ Cost Awareness
- Local embeddings for development
- OpenAI embeddings only for production

---

## 🧪 How to Test

### Positive Test
```
What is Retrieval-Augmented Generation?
```

Expected: Correct, grounded answer.

### Negative Test
```
Who invented RAG?
```

Expected: Explicit refusal.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Sentence-Transformers
- OpenAI (optional)
- PyPDF

---

## 📁 Project Structure

RAG Document Assistant/
├── app.py
├── rag_test_data.txt
├── requirements.txt
├── screenshots/
│   ├── 01_app_home.png
│   ├── 02_pdf_upload.png
│   ├── 03_retrieval.png
│   ├── 04_final_answer.png
│   └── 05_negative_test.png
├── .env (optional)
└── venv/

---

## ▶️ How to Run

```bash
source venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## 🎤 Interview-Ready Explanation

“I implemented a RAG pipeline using FAISS for semantic retrieval and added strict prompt guardrails to eliminate hallucinations. The system supports both local and OpenAI embeddings, enabling cost-efficient development and scalable production deployment.”

---

## 🚀 Future Enhancements

- Source citations per answer
- Confidence scoring
- Multi-document ingestion
- Cloud deployment (AWS / GCP)
- Authentication & access control

---

## 🏁 Final Note

This project demonstrates real-world RAG engineering practices including modular design, hallucination control, evaluation-driven development, and production-ready thinking.
