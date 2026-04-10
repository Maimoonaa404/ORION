import arxiv
import requests
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Free local embeddings — no API key needed
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ✅ EphemeralClient — works on Streamlit Cloud (no disk writes)
chroma_client = chromadb.EphemeralClient()


def fetch_arxiv_papers(query: str, max_results: int = 10):
    """Fetch papers from arXiv"""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in search.results():
        papers.append(Document(
            page_content=f"{result.title}\n\n{result.summary}",
            metadata={
                "title":     result.title,
                "authors":   ", ".join(str(a) for a in result.authors[:3]),
                "url":       result.entry_id,
                "published": str(result.published.date()),
                "source":    "arxiv"
            }
        ))
    return papers


def fetch_semantic_scholar_papers(query: str, max_results: int = 10):
    """Fetch papers from Semantic Scholar API (no API key needed)"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query":  query,
        "limit":  max_results,
        "fields": "title,abstract,authors,year,url"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        papers = []
        for p in response.json().get("data", []):
            abstract = p.get("abstract") or "No abstract available."
            papers.append(Document(
                page_content=f"{p.get('title', '')}\n\n{abstract}",
                metadata={
                    "title":     p.get("title", "Unknown"),
                    "authors":   ", ".join(a["name"] for a in p.get("authors", [])[:3]),
                    "url":       p.get("url", "N/A"),
                    "published": str(p.get("year", "N/A")),
                    "source":    "semantic_scholar"
                }
            ))
        return papers
    except Exception as e:
        print(f"Semantic Scholar error: {e}")
        return []


def fetch_crossref_papers(query: str, max_results: int = 10):
    """Fetch papers from Crossref API (no API key needed)"""
    url = "https://api.crossref.org/works"
    params = {
        "query":  query,
        "rows":   max_results,
        "select": "title,author,published,URL,abstract"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        papers = []
        for item in response.json().get("message", {}).get("items", []):
            title    = item.get("title", ["Unknown"])[0]
            abstract = item.get("abstract", "No abstract available.")
            authors  = ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in item.get("author", [])[:3]
            )
            pub_date = item.get("published", {}).get("date-parts", [["N/A"]])[0]
            papers.append(Document(
                page_content=f"{title}\n\n{abstract}",
                metadata={
                    "title":     title,
                    "authors":   authors or "Unknown",
                    "url":       item.get("URL", "N/A"),
                    "published": str(pub_date[0]),
                    "source":    "crossref"
                }
            ))
        return papers
    except Exception as e:
        print(f"Crossref error: {e}")
        return []


def store_and_retrieve(query: str, papers: list, k: int = 4):
    """Store papers in ChromaDB and retrieve most relevant chunks"""
    vectorstore = Chroma.from_documents(
        documents=papers,
        embedding=embeddings,
        client=chroma_client,
        collection_name="orion_papers"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)
