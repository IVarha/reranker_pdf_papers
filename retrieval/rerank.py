import os
from typing import List, Dict, Any

from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
import torch
from retrieval.embed import Embedder

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/stsb-distilroberta-base"):
        self.model = CrossEncoder(model_name)
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents with their content and metadata
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score required (default 0.0, adjust based on your needs)
            
        Returns:
            List of reranked documents that meet the similarity threshold
        """
        # Prepare document pairs for scoring
        pairs = [(query, doc["content"]) for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Combine documents with their scores
        scored_docs = list(zip(documents, scores))
        
        # Print scores for debugging
        print("Raw scores:", [(float(score), i) for i, (doc, score) in enumerate(scored_docs)])
        
        # Filter by threshold and sort by score in descending order
        filtered_docs = [(doc, score) for doc, score in scored_docs if score >= similarity_threshold]
        filtered_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return [doc for doc, _ in filtered_docs[:top_k]]

def search_and_rerank(query: str, top_k: int = 5):
    """
    Search the vector store and rerank the results.
    
    Args:
        query: The search query
        top_k: Number of results to return
    """
    # Initialize embeddings and vector store
    embeddings = Embedder()
    chroma = Chroma(persist_directory="chroma_db/", embedding_function=embeddings.embeddings)
    
    # Initialize reranker
    reranker = Reranker()
    
    # First retrieve more documents than needed for reranking
    initial_results = chroma.similarity_search_with_score(query, k=top_k * 3)
    
    # Prepare documents for reranking
    documents = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        }
        for doc, score in initial_results
    ]
    
    # Rerank the documents
    reranked_results = reranker.rerank(query, documents, top_k=top_k, similarity_threshold=0.5)
    
    return reranked_results

if __name__ == "__main__":
    # Example usage
    query = "What are the latest developments in active contours?"
    results = search_and_rerank(query, top_k=5)
    
    print(f"\nResults for query: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Content: {result['content'][:200]}...")  # Print first 200 chars
        print(f"Metadata: {result['metadata']}")
        print(f"Initial Score: {result['score']}")
        print("-" * 80)
