import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from retriever import fetch_arxiv_papers, store_and_retrieve
from agents import synthesize_response

load_dotenv()

# Test queries — these are your evaluation dataset
TEST_QUERIES = [
    "What are recent approaches to hallucination reduction in RAG systems?",
    "How do multi-agent systems improve reasoning in LLMs?",
    "What is retrieval augmented generation and how does it work?",
    "What are the challenges of multilingual NLP evaluation?",
    "How do vector databases improve semantic search performance?"
]

def run_evaluation():
    print("🔭 ORION Evaluation Starting...\n")
    
    questions = []
    answers = []
    contexts = []

    for i, query in enumerate(TEST_QUERIES):
        print(f"Running query {i+1}/{len(TEST_QUERIES)}: {query[:50]}...")
        
        try:
            # Fetch and retrieve
            papers = fetch_arxiv_papers(query, max_results=8)
            retrieved_docs = store_and_retrieve(query, papers, k=4)
            
            # Generate response
            response = synthesize_response(query, retrieved_docs)
            
            # Store for evaluation
            questions.append(query)
            answers.append(response)
            contexts.append([doc.page_content for doc in retrieved_docs])
            
            print(f"  ✅ Done\n")
            
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            continue

    # Build RAGAS dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })

    # Wrap Groq LLM and HuggingFace embeddings for RAGAS
    groq_llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    wrapped_llm = LangchainLLMWrapper(groq_llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    print("📊 Running RAGAS evaluation...\n")
    
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings
    )

    print("\n" + "="*50)
    print("🎯 ORION EVALUATION RESULTS")
    print("="*50)
    df = results.to_pandas()
    print(df.to_string())
    print("\n📈 AVERAGE SCORES:")
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            print(f"  {col}: {df[col].mean():.3f}")
    print("="*50)
    
    df.to_csv("orion_evaluation_results.csv", index=False)
    print("\n✅ Results saved to orion_evaluation_results.csv")
    
    return df

if __name__ == "__main__":
    run_evaluation()
