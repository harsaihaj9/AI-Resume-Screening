# 📄 AI Resume Screener

**Screen dozens of resumes against a job description in minutes — not hours.**

An AI-powered recruiting assistant that reads candidate resumes, compares them against your job description requirement-by-requirement, and gives you a ranked, explainable shortlist — all running locally with a free-tier LLM.

---

##  Features

-  **Automatic shortlist generation** — one click ranks every candidate in a Markdown comparison table
-  **Ask anything** — "Who has 5+ years of Python?" / "Does anyone have AWS certifications?"
-  **Explainable scoring** — every candidate gets a match score, strengths, and honest gaps — no black-box answers
-  **Free to run** — local embeddings (no OpenAI billing) + Groq's free-tier LLM
-  **Private by default** — resumes and embeddings stay on your machine, nothing is sent anywhere except the LLM API call itself
-  **Conversation memory** — follow-up questions understand prior context in the same session

---

##  Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| UI | [Streamlit](https://streamlit.io) | Chat-based web interface |
| Orchestration | [LangChain](https://langchain.com) + [LangGraph](https://langchain-ai.github.io/langgraph/) | Agent with tool-calling |
| LLM | [Groq](https://groq.com) — `llama-3.3-70b-versatile` | Reasoning & scoring |
| Embeddings | [HuggingFace](https://huggingface.co) — `sentence-transformers/all-MiniLM-L6-v2` | Free, local resume search |
| Vector Store | LangChain `InMemoryVectorStore` | In-memory semantic search |
| PDF Parsing | `pypdf` | Resume text extraction |

---

## 📁 Project Structure

```
resume-screener/
├── app.py               # Streamlit app — UI, agent, tools
├── requirements.txt      # Python dependencies
├── .env.example           # Template for your API key
├── resume_files/         # Uploaded resumes land here at runtime
└── README.md
```

---

##  Quick Start

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)

### 1. Clone / unzip the project
```bash
cd path/to/resume-screener
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate          
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add your API key
```bash
cat > .env << 'EOF'
GROQ_API_KEY=your_groq_key_here
EOF
```

### 5. Run it
```bash
streamlit run app.py
```

Opens automatically at **http://localhost:8501** 🎉

---

## 🖱️ How to Use

| Step | Action |
|---|---|
| 1️⃣ | Paste the job description into the text box |
| 2️⃣ | Upload one or more candidate resume PDFs |
| 3️⃣ | Click  Start Screening to build the knowledge base |
| 4️⃣ | Click  Auto-generate shortlist for a ranked table, or ask specific questions in chat |
| 5️⃣ | Click  Start over in the sidebar to reset and screen a new batch |

### Sample output

| Candidate | Match Score | Strengths | Gaps | Notes |
|---|---|---|---|---|
| jane_doe_resume.pdf | 8/10 | RAG project, 3 yrs Python, CGPA 9.1 | No mention of C++ | Strong technical fit |
| john_smith_resume.pdf | 5/10 | Good C/C++ background | No RAG or LLM project found | Would need upskilling |





