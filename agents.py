import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import ast
import json
import re

load_dotenv()


GROQ_MODEL = os.getenv("ORION_GROQ_MODEL", "openai/gpt-oss-120b")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=GROQ_MODEL,
    temperature=0.2
)

# Lower temperature for synthesis — improves RAGAS faithfulness (grounding) vs creative drift
llm_synth = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=GROQ_MODEL,
    temperature=0.0
)

# Agent 0 — Router Agent
def classify_query(query: str) -> str:
    """Classifies query as 'research', 'code', or 'both'"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query routing agent.
        Classify the user's query into exactly one of three categories:

        - 'research': questions about concepts, papers, surveys, explanations, comparisons.
          Example: "What is RAG?", "Explain transformers", "How does attention work?"

        - 'code': requests to ONLY generate, write, implement, or debug code with no need for research.
          Example: "Write a Python function to reverse a string", "Debug this code", "Implement a binary search"

        - 'both': requests that need research context AND code implementation together.
          Example: "Explain RAG and show me how to implement it",
                   "What is the attention mechanism and write the code for it",
                   "How does FAISS work and give me a code example"

        Return ONLY one word: research, code, or both.
        Nothing else."""),
        ("human", "Query: {query}")
    ])

    chain = prompt | llm
    result = chain.invoke({"query": query})
    classification = result.content.strip().lower()

    if "both" in classification:
        return "both"
    if "code" in classification:
        return "code"
    return "research"


# Agent 1 — Query Decomposition Agent
def decompose_query(query: str) -> list:
    """Breaks complex query into focused sub-queries"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research query decomposition agent.
        Break the user's research query into 2-3 focused sub-queries
        that together cover the full topic.
        Return ONLY a Python list of strings, nothing else.
        Example: ["sub-query 1", "sub-query 2", "sub-query 3"]"""),
        ("human", "Query: {query}")
    ])

    chain = prompt | llm
    result = chain.invoke({"query": query})

    try:
        sub_queries = ast.literal_eval(result.content.strip())
        return sub_queries
    except:
        return [query]


# Max chars per chunk passed to the LLM — chunks are smaller post-splitter; include enough for RAGAS faithfulness
_SYNTH_CHUNK_CHARS = 1800


# Agent 2 — Synthesis Agent
def synthesize_response(query: str, retrieved_docs: list[Document]) -> str:
    """Synthesizes a cited, grounded response from retrieved papers."""

    if not retrieved_docs:
        return (
            "No passages were retrieved for this query. "
            "Try rephrasing, adding paper sources, or uploading a relevant document."
        )

    context = ""
    for i, doc in enumerate(retrieved_docs):
        body = (doc.page_content or "")[:_SYNTH_CHUNK_CHARS]
        context += f"\n[{i+1}] Title: {doc.metadata.get('title', 'Unknown')}\n"
        context += f"    Authors: {doc.metadata.get('authors', 'Unknown')}\n"
        context += f"    Published: {doc.metadata.get('published', 'Unknown')}\n"
        context += f"    Passage: {body}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research synthesis agent for RAG evaluation.
Rules (strict):
- Use ONLY information that appears in the numbered passages below. Every factual claim must be directly supported by a passage.
- If a claim cannot be supported, omit it or write: "The provided passages do not specify this."
- Do not use outside knowledge, invented citations, or paper titles not present in the passages.
- Answer the user's question directly in the first paragraph (improves answer relevance), then elaborate with citations.
- Cite sources as [1], [2], etc., matching the passage numbers.
- End with a short References list for sources you cited."""),
        ("human", """Research Query: {query}

Retrieved Passages:
{context}

Provide a concise, fully grounded response:""")
    ])

    chain = prompt | llm_synth
    result = chain.invoke({"query": query, "context": context})
    return result.content


# Agent 3 — Quality Check Agent
def check_response_quality(query: str, response: str) -> dict:
    """Evaluates if response actually answers the query"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a quality evaluation agent.
        Evaluate if the response adequately answers the query.
        Return ONLY a JSON object with these exact keys:
        - answers_query: true/false
        - has_citations: true/false
        - confidence: low/medium/high
        No markdown, no code fences, no explanation. Just the raw JSON."""),
        ("human", """Query: {query}
        Response: {response}""")
    ])

    chain = prompt | llm
    result = chain.invoke({"query": query, "response": response})

    raw = result.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        answers    = bool(re.search(r'"answers_query"\s*:\s*true', raw, re.I))
        citations  = bool(re.search(r'"has_citations"\s*:\s*true', raw, re.I))
        conf_match = re.search(r'"confidence"\s*:\s*"(\w+)"', raw, re.I)
        confidence = conf_match.group(1).lower() if conf_match else "medium"
        return {
            "answers_query": answers,
            "has_citations": citations,
            "confidence": confidence if confidence in ("low", "medium", "high") else "medium"
        }


# Agent 4 — Code Stub Generation Agent
def generate_code_stub(query: str, context: str) -> str:
    """Generates a relevant Python code stub based on query and context"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a code generation agent specializing in AI/ML research implementation.
        Based on the query and context provided, generate a clean, well-commented
        Python code stub that demonstrates the core concept.

        Rules:
        - Generate ONLY a Python code stub with comments
        - Include relevant imports
        - Add docstrings explaining what each function does
        - Keep it practical and implementable
        - If the topic is not code-related, return a simple data structure or pseudocode
        - Maximum 50 lines"""),
        ("human", """Query: {query}

Context:
{context}

Generate a Python code stub implementing the core concept:""")
    ])

    chain = prompt | llm
    result = chain.invoke({"query": query, "context": context})
    return result.content


def user_requested_code(query: str) -> bool:
    """Heuristic check for explicit code/file requests in user query."""
    lowered = query.lower()
    code_terms = [
        "code",
        "implement",
        "write",
        "build",
        "script",
        "python",
        "function",
        "class",
        "file",
        "files",
        "api",
    ]
    return any(term in lowered for term in code_terms)


def generate_model_file(query: str, context: str, model_name: str) -> str:
    """Generate model-specific Python file content."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You generate model-specific Python implementation files.
        Return ONLY valid Python code.
        Requirements:
        - Tailor code to the target model/framework name.
        - Add clear function/class names.
        - Keep it concise and runnable.
        - Include brief comments only where needed.
        - Maximum 80 lines."""),
        ("human", """User request: {query}
Target model/framework: {model_name}
Context: {context}

Generate a complete Python file for this target model/framework.""")
    ])

    chain = prompt | llm
    result = chain.invoke({"query": query, "context": context, "model_name": model_name})
    return result.content
