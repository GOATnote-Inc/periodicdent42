"""
Retrieval-Augmented Generation (RAG) System for Scientific Literature

Integrates Vertex AI Vector Search to index and retrieve scientific papers,
providing Gemini with contextual knowledge for experiment planning.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Scientific document representation."""
    id: str
    title: str
    abstract: str
    authors: List[str]
    year: int
    doi: Optional[str] = None
    journal: Optional[str] = None
    keywords: Optional[List[str]] = None
    full_text: Optional[str] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Search result with relevance score."""
    document: Document
    score: float
    rank: int


class RAGSystem:
    """
    RAG system for scientific literature retrieval.
    
    Features:
    - Semantic search using Vertex AI embeddings
    - Multi-field indexing (title, abstract, full text)
    - Citation-aware retrieval
    - Domain-specific filtering
    
    Example usage:
        rag = RAGSystem(project_id="periodicdent42")
        
        # Index papers
        papers = load_papers_from_arxiv()
        rag.index_documents(papers)
        
        # Search
        results = rag.search(
            query="perovskite solar cell stability",
            top_k=5,
            filters={"year": {"$gte": 2020}}
        )
        
        # Use with Gemini
        context = rag.format_context_for_llm(results)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
    """
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        embedding_model: str = "textembedding-gecko@003"
    ):
        self.project_id = project_id
        self.location = location
        self.embedding_model_name = embedding_model
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Load embedding model
        self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model)
        
        # In-memory document store (in production, use Vector Search index)
        self.documents: Dict[str, Document] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        
        logger.info(f"RAG System initialized: {embedding_model}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using Vertex AI.
        
        Args:
            text: Input text (max 3072 tokens for gecko)
        
        Returns:
            Embedding vector (768-dim for gecko)
        """
        # Truncate if too long
        max_length = 3000  # Conservative limit
        if len(text) > max_length:
            text = text[:max_length]
        
        try:
            embeddings = self.embedding_model.get_embeddings([text])
            return np.array(embeddings[0].values)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vector as fallback
            return np.zeros(768)
    
    def embed_batch(self, texts: List[str], batch_size: int = 5) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts
            batch_size: Batch size (Vertex AI limit: 5 per request)
        
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Truncate texts
            batch = [t[:3000] if len(t) > 3000 else t for t in batch]
            
            try:
                embeddings = self.embedding_model.get_embeddings(batch)
                batch_embeddings = [np.array(emb.values) for emb in embeddings]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Add zero vectors for failed batch
                all_embeddings.extend([np.zeros(768) for _ in batch])
        
        return all_embeddings
    
    def index_documents(self, documents: List[Document]) -> int:
        """
        Index documents for semantic search.
        
        Args:
            documents: List of documents to index
        
        Returns:
            Number of documents indexed
        """
        logger.info(f"Indexing {len(documents)} documents...")
        
        # Generate embeddings for all documents
        texts_to_embed = []
        for doc in documents:
            # Combine title and abstract for embedding
            text = f"{doc.title}\n\n{doc.abstract}"
            texts_to_embed.append(text)
        
        embeddings = self.embed_batch(texts_to_embed)
        
        # Store documents and embeddings
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
            self.documents[doc.id] = doc
            self.doc_ids.append(doc.id)
        
        # Create embedding matrix for efficient search
        self.embeddings = np.array([doc.embedding for doc in documents])
        
        logger.info(f"Successfully indexed {len(documents)} documents")
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Search for relevant documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters (e.g., {"year": {"$gte": 2020}})
        
        Returns:
            List of retrieval results sorted by relevance
        """
        if len(self.documents) == 0:
            logger.warning("No documents indexed")
            return []
        
        # Generate query embedding
        query_embedding = self.embed_text(query)
        
        # Compute cosine similarity with all documents
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        
        # Apply filters if provided
        valid_indices = self._apply_filters(filters)
        
        # Get top-k indices
        filtered_similarities = np.array([
            sim if i in valid_indices else -np.inf
            for i, sim in enumerate(similarities)
        ])
        
        top_indices = np.argsort(filtered_similarities)[-top_k:][::-1]
        
        # Create retrieval results
        results = []
        for rank, idx in enumerate(top_indices):
            if filtered_similarities[idx] == -np.inf:
                continue
            
            doc_id = self.doc_ids[idx]
            results.append(RetrievalResult(
                document=self.documents[doc_id],
                score=float(similarities[idx]),
                rank=rank + 1
            ))
        
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
    
    def format_context_for_llm(
        self,
        results: List[RetrievalResult],
        max_length: int = 2000
    ) -> str:
        """
        Format retrieval results as context for LLM.
        
        Args:
            results: Retrieval results
            max_length: Maximum context length in characters
        
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for result in results:
            doc = result.document
            
            # Format citation
            citation = f"[{result.rank}] {doc.title} ({doc.year})"
            if doc.authors:
                authors = ", ".join(doc.authors[:3])
                if len(doc.authors) > 3:
                    authors += " et al."
                citation += f" - {authors}"
            
            # Format abstract
            abstract = doc.abstract[:500] + "..." if len(doc.abstract) > 500 else doc.abstract
            
            entry = f"{citation}\n{abstract}\n"
            entry_length = len(entry)
            
            if current_length + entry_length > max_length:
                break
            
            context_parts.append(entry)
            current_length += entry_length
        
        context = "\n".join(context_parts)
        return context
    
    def _cosine_similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document embeddings."""
        # Normalize vectors
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
        
        # Compute dot product
        similarities = np.dot(doc_norms, query_norm)
        return similarities
    
    def _apply_filters(self, filters: Optional[Dict[str, Any]]) -> set:
        """Apply filters to get valid document indices."""
        if filters is None:
            return set(range(len(self.doc_ids)))
        
        valid_indices = set(range(len(self.doc_ids)))
        
        for field, condition in filters.items():
            if isinstance(condition, dict):
                # Range query (e.g., {"$gte": 2020})
                for op, value in condition.items():
                    for i, doc_id in enumerate(self.doc_ids):
                        doc = self.documents[doc_id]
                        doc_value = getattr(doc, field, None)
                        
                        if doc_value is None:
                            valid_indices.discard(i)
                            continue
                        
                        if op == "$gte" and doc_value < value:
                            valid_indices.discard(i)
                        elif op == "$lte" and doc_value > value:
                            valid_indices.discard(i)
                        elif op == "$eq" and doc_value != value:
                            valid_indices.discard(i)
            else:
                # Equality query
                for i, doc_id in enumerate(self.doc_ids):
                    doc = self.documents[doc_id]
                    if getattr(doc, field, None) != condition:
                        valid_indices.discard(i)
        
        return valid_indices
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        if len(self.documents) == 0:
            return {"num_documents": 0}
        
        years = [doc.year for doc in self.documents.values() if doc.year]
        
        return {
            "num_documents": len(self.documents),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "year_range": (min(years), max(years)) if years else None,
            "avg_abstract_length": np.mean([len(doc.abstract) for doc in self.documents.values()])
        }


# Helper functions for data loading

def load_papers_from_arxiv(
    search_query: str,
    max_results: int = 100
) -> List[Document]:
    """
    Load papers from arXiv API.
    
    Args:
        search_query: Search query (e.g., "cat:cond-mat.mtrl-sci AND perovskite")
        max_results: Maximum number of papers to fetch
    
    Returns:
        List of Document objects
    """
    # TODO: Implement arXiv API integration
    # For now, return dummy data
    logger.warning("arXiv integration not implemented yet - returning dummy data")
    
    dummy_papers = [
        Document(
            id=f"arxiv-{i}",
            title=f"Paper {i}: Advances in Materials Science",
            abstract=f"This paper presents novel findings in materials science research. "
                     f"We demonstrate improved properties through systematic optimization.",
            authors=["Smith, J.", "Doe, A."],
            year=2023,
            doi=f"10.1000/example.{i}",
            journal="Nature Materials",
            keywords=["materials", "optimization"]
        )
        for i in range(min(max_results, 10))
    ]
    
    return dummy_papers


def load_papers_from_pubmed(
    search_query: str,
    max_results: int = 100
) -> List[Document]:
    """
    Load papers from PubMed API.
    
    Args:
        search_query: Search query
        max_results: Maximum number of papers to fetch
    
    Returns:
        List of Document objects
    """
    # TODO: Implement PubMed API integration
    logger.warning("PubMed integration not implemented yet")
    return []

