import os
import ast
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)


# ── Agent 0 — Router ──────────────────────────────────────────────────────────
def classify_query(query: str) -> str:
    """Classifies query as 'research', 'code', or 'both'"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query classification agent.
Classify the user's query into exactly one of these categories:
- research: asking for information, explanations, summaries, comparisons
- code: asking to write, generate, or produce code
- both: asking for both research/explanation AND code

Return ONLY one word: research, code, or both. Nothing else."""),
        ("human", "Query: {query}")
    ])
    chain = prompt | llm
    result = chain.invoke({"query": query})
    label = result.content.strip().lower()
    if label not in ("research", "code", "both"):
        return "research"
    return label


# ── Agent 1 — Query Decomposition ────────────────────────────────────────────
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
        if isinstance(sub_queries, list) and sub_queries:
            return sub_queries
    except Exception:
        pass
    return [query]


# ── Agent 2 — Synthesis ───────────────────────────────────────────────────────
def synthesize_response(query: str, retrieved_docs: list[Document]) -> str:
    """Synthesizes a cited, grounded response from retrieved papers"""
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"\n[{i+1}] Title: {doc.metadata.get('title', 'Unknown')}\n"
        context += f"    Authors: {doc.metadata.get('authors', 'Unknown')}\n"
        context += f"    Published: {doc.metadata.get('published', 'Unknown')}\n"
        context += f"    Content: {doc.page_content[:500]}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research synthesis agent.
Using ONLY the provided papers, answer the research query.
Every claim you make MUST be cited with [1], [2], etc.
If the papers don't contain enough information, say so honestly.
Do NOT hallucinate or add information not in the papers.
End with a References section listing all cited papers."""),
        ("human", """Research Query: {query}

Retrieved Papers:
{context}

Provide a comprehensive, citation-grounded response:""")
    ])
    chain = prompt | llm
    result = chain.invoke({"query": query, "context": context})
    return result.content


# ── Agent 3 — Quality Check ───────────────────────────────────────────────────
def check_response_quality(query: str, response: str) -> dict:
    """Evaluates if response actually answers the query"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a quality evaluation agent.
Evaluate if the response adequately answers the query.
Return ONLY a JSON object with keys:
- answers_query: true or false
- has_citations: true or false
- confidence: low, medium, or high
Nothing else — just the raw JSON, no markdown, no backticks."""),
        ("human", "Query: {query}\nResponse: {response}")
    ])
    chain = prompt | llm
    result = chain.invoke({"query": query, "response": response})
    try:
        return json.loads(result.content.strip())
    except Exception:
        return {"answers_query": True, "has_citations": True, "confidence": "medium"}


# ── Agent 4 — Code Generation ─────────────────────────────────────────────────
def user_requested_code(query: str) -> bool:
    """Returns True if the user's query explicitly asks for code"""
    code_keywords = [
        "code", "implement", "write", "generate", "script",
        "function", "class", "program", "build", "create", "show me how"
    ]
    return any(kw in query.lower() for kw in code_keywords)


def generate_code_stub(query: str, context: str) -> str:
    """Generates a Python code stub based on the query and research context"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a code generation agent.
Based on the research context and query, generate a clean, well-commented
Python code stub or implementation.
Return ONLY the Python code — no explanations outside the code,
use comments inside the code to explain key parts."""),
        ("human", """Query: {query}

Research Context:
{context}

Generate the Python code:""")
    ])
    chain = prompt | llm
    result = chain.invoke({"query": query, "context": context[:2000]})
    # Strip markdown code fences if the model added them
    code = result.content.strip()
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])
    return code.strip()


def generate_model_file(query: str, context: str, model_name: str) -> str:
    """Generates a standalone Python model file for a given model name"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a machine learning code generation agent.
Generate a complete, standalone Python file for the specified model.
Include imports, class definition with __init__ and forward/predict methods,
a brief training loop sketch, and example usage in an if __name__ == '__main__' block.
Use comments to explain design decisions.
Return ONLY the Python code — no markdown fences."""),
        ("human", """Query: {query}
Model to implement: {model_name}

Research Context:
{context}

Generate the complete model file:""")
    ])
    chain = prompt | llm
    result = chain.invoke({
        "query": query,
        "model_name": model_name,
        "context": context[:2000]
    })
    code = result.content.strip()
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])
    return code.strip()
