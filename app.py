import streamlit as st
import fitz  # PyMuPDF
import re
from langchain_core.documents import Document
from retriever import (
    fetch_arxiv_papers,
    fetch_semantic_scholar_papers,
    fetch_crossref_papers,
    store_and_retrieve,
)
from agents import (
    classify_query,
    decompose_query,
    synthesize_response,
    check_response_quality,
    generate_code_stub,
    user_requested_code,
)

st.set_page_config(
    page_title="ORION – Agentic Research Assistant",
    page_icon="🔭",
    layout="wide"
)

st.title("🔭 ORION")
st.subheader("Agentic AI Research Assistant")
st.markdown("*Multi-agent system for structured, citation-grounded research*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    max_papers = st.slider("Papers to retrieve", 5, 20, 10)
    sources = st.multiselect(
        "Paper sources",
        options=["arXiv", "Semantic Scholar", "Crossref"],
        default=["arXiv"],
        help="Select one or more paper databases."
    )
    show_sources = st.checkbox("Show source papers", value=True)
    show_agents  = st.checkbox("Show agent reasoning", value=True)
    st.divider()

    # ── Document Upload ───────────────────────────────────────────────────────
    st.header("📎 Upload Documents")
    st.markdown("Upload PDFs or text files to include in retrieval alongside arXiv papers.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Uploaded content is added to the retrieval context."
    )

    uploaded_docs = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.type == "application/pdf":
                    pdf_bytes = uploaded_file.read()
                    pdf_doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
                    text = ""
                    for page in pdf_doc:
                        text += page.get_text()
                    pdf_doc.close()
                else:
                    text = uploaded_file.read().decode("utf-8", errors="ignore")

                if text.strip():
                    uploaded_docs.append(Document(
                        page_content=text[:3000],
                        metadata={
                            "title":     uploaded_file.name,
                            "authors":   "Uploaded by user",
                            "url":       "Local upload",
                            "published": "N/A",
                            "source":    "upload"
                        }
                    ))
                    st.success(f"✅ {uploaded_file.name}")
                else:
                    st.warning(f"⚠️ {uploaded_file.name} appears empty.")

            except Exception as e:
                st.error(f"❌ Could not read {uploaded_file.name}: {e}")

        if uploaded_docs:
            st.info(f"📄 {len(uploaded_docs)} document(s) ready.")

    st.divider()
    st.markdown("**Stack:**")
    st.markdown("- LangChain orchestration")
    st.markdown("- ChromaDB vector store")
    st.markdown("- Multi-source paper retrieval")
    st.markdown("- Groq / LLaMA 3")

# ── Main Input ────────────────────────────────────────────────────────────────
query = st.text_area(
    "Enter your research query:",
    placeholder="e.g. What are the latest approaches to hallucination reduction in RAG systems?",
    height=100
)

if st.button("🚀 Run ORION", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a research query.")
    elif not sources and not uploaded_docs:
        st.warning("Select at least one paper source or upload a document.")
    else:
        response   = ""
        code_stub  = ""
        quality    = {}
        retrieved  = []
        wants_code = user_requested_code(query)

        # ── Agent 0 — Router ──────────────────────────────────────────────────
        with st.status("🧭 Agent 0: Classifying query...", expanded=show_agents) as status:
            query_type = classify_query(query)
            label_map  = {
                "research": "🔬 Research",
                "code":     "💻 Code",
                "both":     "🔬💻 Research + Code"
            }
            st.write(f"**Query classified as:** `{label_map.get(query_type, query_type)}`")
            status.update(label=f"✅ Agent 0: Routed as '{query_type}'", state="complete")

        # ══════════════════════════════════════════════════════════════════════
        # ROUTE: research or both — fetch papers and synthesize
        # ══════════════════════════════════════════════════════════════════════
        if query_type in ("research", "both"):

            # Agent 1 — Decomposition
            with st.status("🤖 Agent 1: Decomposing query...", expanded=show_agents) as status:
                sub_queries = decompose_query(query)
                st.write("**Sub-queries generated:**")
                for i, sq in enumerate(sub_queries):
                    st.write(f"{i+1}. {sq}")
                status.update(label="✅ Agent 1: Query decomposed", state="complete")

            # Retrieval
            with st.status("📚 Retrieving papers...", expanded=False) as status:
                all_papers = []
                for sq in sub_queries:
                    per_query_limit = max(1, max_papers // len(sub_queries))
                    if "arXiv" in sources:
                        all_papers.extend(fetch_arxiv_papers(sq, max_results=per_query_limit))
                    if "Semantic Scholar" in sources:
                        all_papers.extend(fetch_semantic_scholar_papers(sq, max_results=per_query_limit))
                    if "Crossref" in sources:
                        all_papers.extend(fetch_crossref_papers(sq, max_results=per_query_limit))

                if uploaded_docs:
                    all_papers = uploaded_docs + all_papers

                if not all_papers:
                    retrieved = []
                    status.update(label="⚠️ No papers found for selected sources", state="complete")
                else:
                    retrieved = store_and_retrieve(query, all_papers)

                upload_count   = sum(1 for d in retrieved if d.metadata.get("source") == "upload")
                external_count = len(retrieved) - upload_count
                label = f"✅ Retrieved {len(retrieved)} sources"
                if upload_count:
                    label += f" ({upload_count} uploaded, {external_count} external)"
                if all_papers:
                    status.update(label=label, state="complete")

            # Agent 2 — Synthesis
            with st.status("✍️ Agent 2: Synthesizing response...", expanded=False) as status:
                response = synthesize_response(query, retrieved)
                status.update(label="✅ Agent 2: Response synthesized", state="complete")

            # Agent 3 — Quality Check
            with st.status("🔍 Agent 3: Checking quality...", expanded=False) as status:
                quality = check_response_quality(query, response)
                status.update(label="✅ Agent 3: Quality verified", state="complete")

            # Agent 4 — Code Stub (only on explicit user request)
            if wants_code:
                with st.status("💻 Agent 4: Generating code stub...", expanded=False) as status:
                    code_stub = generate_code_stub(query, response)
                    status.update(label="✅ Agent 4: Code stub generated", state="complete")

        # ══════════════════════════════════════════════════════════════════════
        # ROUTE: code only — skip retrieval entirely
        # ══════════════════════════════════════════════════════════════════════
        elif query_type == "code":
            with st.status("💻 Agent 4: Generating code directly...", expanded=show_agents) as status:
                code_stub = generate_code_stub(query, query)
                response  = ""
                quality   = {"answers_query": True, "has_citations": False, "confidence": "high"}
                status.update(label="✅ Code generated", state="complete")

        # ── Results ───────────────────────────────────────────────────────────
        st.divider()

        # Quality badges (skip for pure code)
        if query_type != "code":
            col1, col2, col3 = st.columns(3)
            with col1:
                if quality.get("answers_query"):
                    st.success("✅ Answers query")
                else:
                    st.error("❌ Partial answer")
            with col2:
                if quality.get("has_citations"):
                    st.success("✅ Citations present")
                else:
                    st.warning("⚠️ Limited citations")
            with col3:
                confidence = quality.get("confidence", "medium")
                if confidence == "high":
                    st.success(f"✅ Confidence: {confidence}")
                elif confidence == "medium":
                    st.warning(f"⚠️ Confidence: {confidence}")
                else:
                    st.error(f"❌ Confidence: {confidence}")

        # Research Summary
        if response:
            st.divider()
            st.markdown("### 📋 Research Summary")
            st.markdown(response)

        # Code Stub
        if code_stub and wants_code:
            st.divider()
            st.markdown("### 💻 Code Stub")
            st.caption("Auto-generated Python stub based on the query above.")
            st.code(code_stub, language="python")

        # Source Papers
        if show_sources and retrieved:
            st.divider()
            st.markdown("### 📄 Source Papers")
            for i, doc in enumerate(retrieved):
                source_name = doc.metadata.get("source", "unknown")
                source_labels = {
                    "upload":           "📎 Uploaded",
                    "arxiv":            "🔬 arXiv",
                    "semantic_scholar": "🧠 Semantic Scholar",
                    "crossref":         "📚 Crossref",
                }
                source_tag = source_labels.get(source_name, f"📄 {source_name}")
                with st.expander(f"[{i+1}] {source_tag} — {doc.metadata.get('title', 'Unknown')}"):
                    st.write(f"**Authors:** {doc.metadata.get('authors', 'Unknown')}")
                    st.write(f"**Published:** {doc.metadata.get('published', 'Unknown')}")
                    if doc.metadata.get("url") != "Local upload":
                        st.write(f"**URL:** {doc.metadata.get('url', 'N/A')}")
                    st.write(f"**Abstract:** {doc.page_content[:400]}...")
