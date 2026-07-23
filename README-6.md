# 📄 AI Resume Screener

A Retrieval-Augmented Generation (RAG) app that lets recruiters upload candidate resumes (PDFs), paste a job description, and chat with an AI agent to screen, rank, and shortlist candidates — all grounded in the actual resume content.

Built with **Streamlit**, **LangChain**, **LangGraph**, and **Groq**.

---

## How it works

1. You paste a job description and upload one or more candidate resumes (PDF).
2. Each resume is split into chunks, embedded using OpenAI embeddings, and stored in an in-memory vector store.
3. An AI agent (powered by Groq's `openai/gpt-oss-20b` model) uses a retrieval tool to pull relevant resume excerpts and answer your questions or generate a ranked shortlist — grounded strictly in the retrieved content, not guesses.
4. You can either click **"Auto-generate shortlist"** for a full ranked comparison of all candidates, or ask specific questions like *"Who has 5+ years of Python experience?"*

---

## Features

- 📥 Upload multiple resumes at once
- 📝 Paste a job description that the agent evaluates every candidate against
- 🔍 Retrieval tool that cites which resume (filename) each excerpt came from
- 🏆 One-click auto-shortlist with match scores (out of 10) and justifications
- 💬 Follow-up chat for ad-hoc candidate questions
- 🧠 Conversation memory via LangGraph's checkpointer (per session)

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```
- `OPENAI_API_KEY` is used for generating embeddings (`text-embedding-3-large`)
- `GROQ_API_KEY` is used for the LLM (`openai/gpt-oss-20b` via Groq)

### 5. Run the app
```bash
streamlit run resume_screener.py
```
The app will open in your browser at `http://localhost:8501`.

---

## Project structure
```
.
├── resume_screener.py   # Main Streamlit app
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not committed — see .gitignore)
├── .gitignore
└── README.md
```

---

## Notes & limitations

- Resumes are stored in-memory for the session only (`InMemoryVectorStore`); nothing persists once the app restarts.
- Uploaded PDF files are temporarily saved to a local `resume_files/` folder — make sure this is git-ignored if resumes contain personal data (PII).
- The vector store and job description are scoped to a single screening session; to screen a new batch of candidates, restart the app.
- This is a demo/prototype — for production use with real candidate PII, consider adding authentication, persistent/secure storage, and data-retention controls in line with applicable privacy regulations.

---

## License

MIT — feel free to use, modify, and share.
