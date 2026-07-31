from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
defaults = {
    "resumes_uploaded": False,
    "agent": None,
    "vector_store": None,
    "messages": [],
    "job_description": "",
    "candidate_count": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------------------------------
# Core processing
# --------------------------------------------------------------------------
def process_resumes(path, job_description):
    """Load resumes, build the vector store, and construct the screening agent."""

    # load all resumes from folder
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # tag each chunk with its source filename so the agent can cite candidates
    for doc in docs:
        doc.metadata["candidate_file"] = os.path.basename(doc.metadata.get("source", "unknown"))

    candidate_files = sorted({doc.metadata["candidate_file"] for doc in docs})
    st.session_state.candidate_count = len(candidate_files)

    # split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents=docs)

    # embeddings and vector DB (free, runs locally — no API key or billing needed)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = InMemoryVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
    )
    st.session_state.vector_store = vector_db

    # LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    @tool
    def retrieve_resume_context(query: str):
        """Retrieve resume excerpts relevant to a query (skills, experience, education, etc.)
        from the uploaded candidate resumes. Returns excerpts along with the source filename
        so candidates can be identified."""
        results = vector_db.similarity_search(query=query, k=5)
        chunks = []
        for doc in results:
            candidate = doc.metadata.get("candidate_file", "unknown")
            chunks.append(f"[Candidate file: {candidate}]\n{doc.page_content}")
        return "\n\n---\n\n".join(chunks) if chunks else "No relevant resume content found."

    @tool
    def list_candidates():
        """List the distinct candidate resume filenames available in the knowledge base."""
        return "\n".join(candidate_files)

    @tool
    def get_candidate_full_text(candidate_file: str):
        """Retrieve the COMPLETE resume text for one specific candidate (use the exact
        filename from list_candidates). Use this whenever you need to check a candidate
        against multiple requirements at once (e.g. ranking/shortlisting), since a
        similarity search alone can miss relevant details buried in their resume."""
        matched = [
            doc.page_content for doc in split_docs
            if doc.metadata.get("candidate_file") == candidate_file
        ]
        return "\n".join(matched) if matched else f"No content found for '{candidate_file}'."

    system_prompt = f"""You are an AI recruiting assistant that screens candidate resumes against a job description.

Job description:
\"\"\"
{job_description}
\"\"\"

Your knowledge base consists of resumes uploaded by the recruiter (one or more PDF files).
Use `list_candidates` to see which candidates are available. Use `retrieve_resume_context`
for narrow, specific questions (e.g. "who has AWS experience?"). Use `get_candidate_full_text`
whenever you need the COMPLETE picture of one candidate — this is REQUIRED whenever you are
screening, ranking, comparing, or shortlisting, because a similarity search alone can miss
relevant details (like a specific project or skill) buried in a candidate's resume.

When asked to screen, rank, or shortlist candidates:
1. Call `list_candidates` to get every candidate filename.
2. Call `get_candidate_full_text` separately for EACH candidate filename before scoring anyone —
   never rely only on `retrieve_resume_context` for this task.
3. Extract the specific requirements from the job description (skills, experience, education,
   tools, project types, etc.) as a checklist.
4. For each candidate, check them against EVERY item in that checklist individually — don't
   just skim for a general impression.
5. Present the final result as a **Markdown table**, ranked best to worst, with these exact columns:
   | Candidate (filename) | Match Score (/10) | Strengths | Gaps / Missing | Notes |
6. After the table, add a short paragraph per candidate (2-4 sentences) explaining the score in
   plain language — reference specific resume content (projects, skills, years of experience)
   rather than vague statements like "good fit."
7. Be honest about gaps; do not inflate scores. If a requirement isn't mentioned anywhere in a
   candidate's resume, say so explicitly rather than assuming it.
8. Never fabricate candidate details that aren't present in the retrieved context.

For any other question (not a full screen/rank request), answer directly and concisely, citing
the source filename, without forcing a table unless the recruiter asks for a comparison.
"""

    memory = InMemorySaver()
    agent = create_agent(
        model=llm,
        tools=[retrieve_resume_context, list_candidates, get_candidate_full_text],
        system_prompt=system_prompt,
        checkpointer=memory,
    )
    st.session_state.agent = agent
    st.session_state.resumes_uploaded = True


def reset_app():
    for key, value in defaults.items():
        st.session_state[key] = value


def run_agent(query: str) -> str:
    response = st.session_state.agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}},
    )
    return response["messages"][-1].content


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📄 AI Resume Screener")
st.caption("Upload a job description and candidate resumes, then chat with an AI recruiting assistant.")

with st.sidebar:
    st.header("⚙️ Session")
    if st.session_state.resumes_uploaded:
        st.success(f"{st.session_state.candidate_count} candidate(s) loaded")
        st.write("**Job description in use:**")
        st.caption(st.session_state.job_description[:300] + ("…" if len(st.session_state.job_description) > 300 else ""))
        if st.button("🔄 Start over", use_container_width=True):
            reset_app()
            st.rerun()
    else:
        st.info("Upload a job description and resumes to begin.")

    st.divider()
    st.caption("Environment variable required: `GROQ_API_KEY` (see `.env`). Embeddings run locally, free.")

missing_keys = [k for k in ("GROQ_API_KEY",) if not os.getenv(k)]
if missing_keys:
    st.warning(
        f"Missing environment variable(s): {', '.join(missing_keys)}. "
        "Add them to a `.env` file in the project root before screening resumes."
    )

# --------------------------------------------------------------------------
# Step 1: Setup (job description + resume upload)
# --------------------------------------------------------------------------
if not st.session_state.resumes_uploaded:
    st.subheader("1. Paste the job description")
    jd_input = st.text_area("Job description", height=200, placeholder="Paste the JD here...")

    st.subheader("2. Upload candidate resumes")
    uploaded = st.file_uploader(label="Select Resume PDFs", type=["pdf"], accept_multiple_files=True)

    start_disabled = not (jd_input and uploaded) or bool(missing_keys)
    if st.button("🚀 Start Screening", disabled=start_disabled, type="primary"):
        with st.spinner("Processing resumes and building knowledge base…"):
            path = "./resume_files/"
            os.makedirs(path, exist_ok=True)
            # clear any resumes from a previous session
            for f in os.listdir(path):
                fp = os.path.join(path, f)
                if os.path.isfile(fp):
                    os.remove(fp)
            for file in uploaded:
                with open(os.path.join(path, file.name), "wb") as f:
                    f.write(file.getvalue())
            st.session_state.job_description = jd_input
            process_resumes(path, jd_input)
        st.rerun()

# --------------------------------------------------------------------------
# Step 2: Chat UI
# --------------------------------------------------------------------------
if st.session_state.resumes_uploaded and st.session_state.agent:
    with st.expander("📋 Job Description", expanded=False):
        st.write(st.session_state.job_description)

    col1, col2 = st.columns([1, 4])
    with col1:
        auto_clicked = st.button("🔍 Auto-generate shortlist")

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    query = st.chat_input("Ask about candidates, e.g. 'Who has 5+ years of Python experience?'")

    if auto_clicked:
        query = (
            "Screen all candidates in the knowledge base against the job description. "
            "Get each candidate's full resume text first, then rank them best to worst "
            "in a Markdown comparison table, followed by a short explanation per candidate."
        )

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Screening candidates…"):
                answer = run_agent(query)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
