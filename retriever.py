import arxiv
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ✅ Changed from PersistentClient → EphemeralClient
chroma_client = chromadb.EphemeralClient()

def fetch_arxiv_papers(query: str, max_results: int = 10):
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
                "title": result.title,
                "authors": ", ".join(str(a) for a in result.authors[:3]),
                "url": result.entry_id,
                "published": str(result.published.date())
            }
        ))
    return papers

def store_and_retrieve(query: str, papers: list, k: int = 4):
    vectorstore = Chroma.from_documents(
        documents=papers,
        embedding=embeddings,
        client=chroma_client,
        collection_name="orion_papers"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)
