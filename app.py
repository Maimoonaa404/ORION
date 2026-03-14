import streamlit as st
from retriever import fetch_arxiv_papers, store_and_retrieve
from agents import decompose_query, synthesize_response, check_response_quality

st.set_page_config(
    page_title="ORION – Agentic Research Assistant",
    page_icon="🔭",
    layout="wide"
)

st.title("🔭 ORION")
st.subheader("Agentic AI Research Assistant")
st.markdown("*Multi-agent system for structured, citation-grounded research*")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_papers = st.slider("Papers to retrieve", 5, 20, 10)
    show_sources = st.checkbox("Show source papers", value=True)
    show_agents = st.checkbox("Show agent reasoning", value=True)
    st.divider()
    st.markdown("**Stack:**")
    st.markdown("- LangChain orchestration")
    st.markdown("- ChromaDB vector store")
    st.markdown("- arXiv paper retrieval")
    st.markdown("- Groq / LLaMA 3")

# Main input
query = st.text_area(
    "Enter your research query:",
    placeholder="e.g. What are the latest approaches to hallucination reduction in RAG systems?",
    height=100
)

if st.button("🚀 Run ORION", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a research query.")
    else:
        # Agent 1 — Query Decomposition
        with st.status("🤖 Agent 1: Decomposing query...", expanded=show_agents) as status:
            sub_queries = decompose_query(query)
            st.write("**Sub-queries generated:**")
            for i, sq in enumerate(sub_queries):
                st.write(f"{i+1}. {sq}")
            status.update(label="✅ Agent 1: Query decomposed", state="complete")

        # Retrieval
        with st.status("📚 Retrieving papers from arXiv...", expanded=False) as status:
            all_papers = []
            for sq in sub_queries:
                papers = fetch_arxiv_papers(sq, max_results=max_papers//len(sub_queries))
                all_papers.extend(papers)
            
            retrieved = store_and_retrieve(query, all_papers)
            status.update(
                label=f"✅ Retrieved {len(retrieved)} relevant papers",
                state="complete"
            )

        # Agent 2 — Synthesis
        with st.status("✍️ Agent 2: Synthesizing response...", expanded=False) as status:
            response = synthesize_response(query, retrieved)
            status.update(label="✅ Agent 2: Response synthesized", state="complete")

        # Agent 3 — Quality Check
        with st.status("🔍 Agent 3: Checking quality...", expanded=False) as status:
            quality = check_response_quality(query, response)
            status.update(label="✅ Agent 3: Quality verified", state="complete")

        st.divider()

        # Quality badge
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

        # Response
        st.divider()
        st.markdown("### 📋 Research Summary")
        st.markdown(response)

        # Sources
        if show_sources:
            st.divider()
            st.markdown("### 📄 Source Papers")
            for i, doc in enumerate(retrieved):
                with st.expander(f"[{i+1}] {doc.metadata.get('title', 'Unknown')}"):
                    st.write(f"**Authors:** {doc.metadata.get('authors', 'Unknown')}")
                    st.write(f"**Published:** {doc.metadata.get('published', 'Unknown')}")
                    st.write(f"**URL:** {doc.metadata.get('url', 'Unknown')}")
                    st.write(f"**Abstract:** {doc.page_content[:400]}...")
