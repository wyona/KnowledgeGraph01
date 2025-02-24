"""Hybrid search implementation combining Neo4j and FAISS.

This module provides functionality for:
- Vector similarity search using FAISS
- Graph traversal algorithms
- Hybrid ranking combining both approaches
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

# Search result types
class SearchResult:
    """Container for search results with relevance scores."""
    
    def __init__(
        self,
        node_id: str,
        score: float,
        properties: Dict,
        path: Optional[List[Dict]] = None
    ):
        """Initialize search result.
        
        Args:
            node_id: Neo4j node ID
            score: Combined relevance score (0-1)
            properties: Node properties
            path: Optional path of nodes/relationships to this result
        """
        self.node_id = node_id
        self.score = score
        self.properties = properties
        self.path = path or []
        
    def to_dict(self) -> Dict:
        """Convert search result to dictionary format."""
        return {
            "node_id": self.node_id,
            "score": self.score,
            "properties": self.properties,
            "path": self.path
        }

class SearchResults:
    """Container for multiple search results with metadata."""
    
    def __init__(
        self,
        results: List[SearchResult],
        total_found: int,
        query_time_ms: float
    ):
        """Initialize search results.
        
        Args:
            results: List of SearchResult objects
            total_found: Total number of matches found
            query_time_ms: Query execution time in milliseconds
        """
        self.results = results
        self.total_found = total_found
        self.query_time_ms = query_time_ms
        
    def to_dict(self) -> Dict:
        """Convert search results to dictionary format."""
        return {
            "results": [r.to_dict() for r in self.results],
            "total_found": self.total_found,
            "query_time_ms": self.query_time_ms
        }

# Search parameters
class SearchParams:
    """Parameters for configuring hybrid search behavior."""
    
    def __init__(
        self,
        vector_weight: float = 0.5,
        max_results: int = 10,
        min_score: float = 0.0,
        include_paths: bool = False,
        filters: Optional[Dict] = None
    ):
        """Initialize search parameters.
        
        Args:
            vector_weight: Weight of vector similarity vs graph score (0-1)
            max_results: Maximum number of results to return
            min_score: Minimum score threshold for results
            include_paths: Whether to include graph paths to results
            filters: Optional filters to apply (e.g. entity type, date range)
        """
        self.vector_weight = vector_weight
        self.max_results = max_results
        self.min_score = min_score
        self.include_paths = include_paths
        self.filters = filters or {}
        
    def to_dict(self) -> Dict:
        """Convert parameters to dictionary format."""
        return {
            "vector_weight": self.vector_weight,
            "max_results": self.max_results,
            "min_score": self.min_score,
            "include_paths": self.include_paths,
            "filters": self.filters
        }

# Utility functions
def combine_scores(
    vector_scores: np.ndarray,
    graph_scores: np.ndarray,
    weight: float = 0.5
) -> np.ndarray:
    """Combine vector similarity and graph relevance scores.
    
    Args:
        vector_scores: Array of vector similarity scores
        graph_scores: Array of graph relevance scores
        weight: Weight of vector scores vs graph scores (0-1)
        
    Returns:
        Array of combined scores
    """
    # Normalize score arrays
    v_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min())
    g_scores = (graph_scores - graph_scores.min()) / (graph_scores.max() - graph_scores.min())
    
    # Weighted combination
    return weight * v_scores + (1 - weight) * g_scores
