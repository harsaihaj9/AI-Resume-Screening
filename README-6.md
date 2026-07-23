# AI Resume Screener

A RAG-based recruiting assistant. Upload candidate resumes and a job description, and chat with an AI agent that ranks and screens candidates using retrieval-augmented generation.

Built with Streamlit, LangChain, LangGraph, and Groq.

## How it works

1. Paste a job description and upload candidate resumes (PDF).
2. Resumes are chunked, embedded with OpenAI embeddings, and stored in an in-memory vector store.
3. An AI agent retrieves relevant resume excerpts and answers questions or generates a ranked shortlist, grounded in the retrieved content.
4. Click "Auto-generate shortlist" for a full ranked comparison, or ask specific questions like "Who has 5+ years of Python experience?"

## Features

- Upload multiple resumes at once
- Job-description-aware screening with match scores and justifications
- Retrieval tool that cites the source resume for every claim
- Follow-up chat for ad-hoc candidate questions
- Session memory via LangGraph's checkpointer

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

`OPENAI_API_KEY` is used for embeddings (`text-embedding-3-large`). `GROQ_API_KEY` is used for the LLM (`openai/gpt-oss-20b` via Groq).

Run the app:

```bash
streamlit run resume_screener.py
```

## Notes

- Resumes and embeddings are stored in memory only; nothing persists after the app restarts.
- To screen a new batch of candidates, restart the app.
- This is a prototype; a production version would need persistent storage, authentication, and data-retention controls.


