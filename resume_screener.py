from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
import os

st.set_page_config(page_title="AI Resume Screener", layout="wide")

### data in st session
if "resumes_uploaded" not in st.session_state:
    st.session_state.resumes_uploaded = False
if "agent" not in st.session_state:
    st.session_state.agent = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "job_description" not in st.session_state:
    st.session_state.job_description = ""


def process_resumes(path, job_description):
    ## load all resumes from folder
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    ## tag each chunk with its source filename so the agent can cite candidates
    for doc in docs:
        doc.metadata["candidate_file"] = os.path.basename(doc.metadata.get("source", "unknown"))

    ## split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents=docs)

    ## embeddings and Vector DB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )
    st.session_state.vector_store = vector_db

    ## create agent - tool, llm, prompt
    llm = ChatGroq(model="openai/gpt-oss-20b")

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
        files = sorted({doc.metadata.get("candidate_file", "unknown") for doc in docs})
        return "\n".join(files)

    system_prompt = f"""You are an AI recruiting assistant that screens candidate resumes against a job description.

Job description:
\"\"\"
{job_description}
\"\"\"

Your knowledge base consists of resumes uploaded by the recruiter (one or more PDF files).
ALWAYS use the `retrieve_resume_context` tool to pull relevant resume excerpts before answering
questions about candidates, skills, or experience. Use `list_candidates` if you need to know
which candidates are available.

When asked to screen, rank, or shortlist candidates:
- Evaluate each candidate against the job description's required skills, experience, and qualifications.
- Give each candidate a match score out of 10 with a short justification.
- Clearly state the candidate's source filename so the recruiter knows which resume it is.
- Be honest about gaps or missing qualifications; do not inflate scores.
- Never fabricate candidate details that aren't present in the retrieved context.
"""

    memory = InMemorySaver()
    agent = create_agent(
        model=llm,
        tools=[retrieve_resume_context, list_candidates],
        system_prompt=system_prompt,
        checkpointer=memory
    )
    st.session_state.agent = agent
    st.session_state.resumes_uploaded = True


### upload ui
st.title("📄 AI Resume Screener")

if not st.session_state.resumes_uploaded:
    st.subheader("1. Paste the job description")
    jd_input = st.text_area("Job description", height=200, placeholder="Paste the JD here...")

    st.subheader("2. Upload candidate resumes")
    uploaded = st.file_uploader(label="Select Resume PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Start Screening", disabled=not (jd_input and uploaded)):
        with st.spinner("Processing resumes and building knowledge base..."):
            path = "./resume_files/"
            os.makedirs(path, exist_ok=True)
            for file in uploaded:
                with open(os.path.join(path, file.name), "wb") as f:
                    f.write(file.getvalue())
            st.session_state.job_description = jd_input
            process_resumes(path, jd_input)
            st.rerun()

## chat ui
if st.session_state.resumes_uploaded and st.session_state.agent:
    with st.expander("Job Description", expanded=False):
        st.write(st.session_state.job_description)

    if st.button("🔍 Auto-generate shortlist"):
        auto_query = "Screen all candidates in the knowledge base against the job description. Rank them by match score from best to worst, with justification for each."
        st.session_state.messages.append({"role": "user", "content": auto_query})

    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask about candidates, e.g. 'Who has 5+ years of Python experience?'")
    pending = query or (
        st.session_state.messages[-1]["content"]
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user"
        and (len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] != "user")
        else None
    )

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": 1}}
        )
        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role": "ai", "content": answer})

    # handle the auto-generated shortlist query if it hasn't been answered yet
    if (
        st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
        and not query
    ):
        last_query = st.session_state.messages[-1]["content"]
        with st.chat_message("ai"):
            with st.spinner("Screening candidates..."):
                response = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": last_query}]},
                    {"configurable": {"thread_id": 1}}
                )
                answer = response["messages"][-1].content
                st.markdown(answer)
        st.session_state.messages.append({"role": "ai", "content": answer})
