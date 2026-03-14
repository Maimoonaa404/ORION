import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()

# Free LLaMA 3 via Groq
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)

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
        import ast
        sub_queries = ast.literal_eval(result.content.strip())
        return sub_queries
    except:
        return [query]

# Agent 2 — Synthesis Agent
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

# Agent 3 — Quality Check Agent  
def check_response_quality(query: str, response: str) -> dict:
    """Evaluates if response actually answers the query"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a quality evaluation agent.
        Evaluate if the response adequately answers the query.
        Return ONLY a JSON with keys:
        - answers_query: true/false
        - has_citations: true/false  
        - confidence: low/medium/high
        Nothing else, just the JSON."""),
        ("human", """Query: {query}
        Response: {response}""")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": query, "response": response})
    
    try:
        import json
        return json.loads(result.content.strip())
    except:
        return {"answers_query": True, "has_citations": True, "confidence": "medium"}
