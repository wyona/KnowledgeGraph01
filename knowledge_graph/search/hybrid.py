"""Hybrid search implementation combining vector similarity and graph traversal."""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from neo4j import GraphDatabase

from . import SearchResult, SearchResults, SearchParams, combine_scores
from .vector_store import VectorStore

class HybridSearchEngine:
    """Search engine combining FAISS and Neo4j capabilities."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str
    ):
        """Initialize the hybrid search engine.
        
        Args:
            vector_store: Initialized VectorStore instance
            neo4j_uri: Neo4j database URI
            neo4j_user: Database username
            neo4j_password: Database password
        """
        self.vector_store = vector_store
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )

    def search(
        self,
        query_vector: np.ndarray,
        params: Optional[SearchParams] = None
    ) -> SearchResults:
        """Perform hybrid search using vector similarity and graph traversal.
        
        Args:
            query_vector: Query vector for similarity search
            params: Search parameters to configure behavior
            
        Returns:
            SearchResults containing ranked matches
        """
        start_time = time.time()
        
        # Use default parameters if none provided
        print(f"Search parameters: {params}")
        params = params or SearchParams()
        
        # Get initial candidates from vector search
        print("Get initial candidates from vector search ...")
        #print(f"Search vector store using embedding '{query_vector}'")
        distances, indices, node_ids = self.vector_store.search(
            query_vector,
            k=params['max_results'] * 2  # Get extra candidates for reranking
        )

        for node_id in node_ids:
            print(f"Initial candidate found: {node_id}")
        
        # Convert distances to similarity scores (1 - normalized distance)
        vector_scores = 1 - (distances - distances.min()) / (distances.max() - distances.min())

        for score in vector_scores:
            print(f"Vector score: {score}")
        
        # Get graph-based scores for candidates
        print("Get graph-based scores for candidates ...")
        graph_scores = self._get_graph_scores(
            node_ids,
            params['filters'],
            params['include_paths']
        )

        for score in graph_scores:
            print(f"Graph score: {score}")
        
        # Combine scores
        combined_scores = combine_scores(
            vector_scores,
            graph_scores,
            params['vector_weight']
        )

        for score in combined_scores:
            print(f"Combined score: {score}")
        
        # Sort by combined score
        sorted_indices = np.argsort(combined_scores)[::-1]
        
        # Build results
        results = []
        for idx in sorted_indices:
            score = combined_scores[idx]
            if score < params['min_score']:
                print(f"Score {score} below min score.")
                break
                
            if len(results) >= params['max_results']:
                print(f"results above max results: {len(results)}")
                break
                
            node_id = node_ids[idx]
            if node_id is None:
                continue
                
            # Get node properties and path
            with self.neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (n)
                    WHERE n.id = $node_id
                    RETURN n
                    """,
                    node_id=node_id
                )
                record = result.single()
                if not record:
                    continue
                    
                properties = dict(record["n"])
                
                # Get path if requested
                path = None
                if params['include_paths']:
                    path = self._get_path_to_node(node_id)
                
                results.append(SearchResult(
                    node_id=node_id,
                    score=float(score),
                    properties=properties,
                    path=path
                ))
        
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return SearchResults(
            results=results,
            total_found=len(results),
            query_time_ms=query_time
        )

    def _get_graph_scores(
        self,
        node_ids: List[str],
        filters: Dict,
        include_paths: bool
    ) -> np.ndarray:
        """Calculate graph-based relevance scores for nodes.
        
        Args:
            node_ids: List of candidate node IDs
            filters: Query filters to apply
            include_paths: Whether to include path information
            
        Returns:
            Array of graph relevance scores
        """
        scores = np.zeros(len(node_ids))
        
        with self.neo4j_driver.session() as session:
            for i, node_id in enumerate(node_ids):
                if node_id is None:
                    print(f"Graph does not contain node '{node_id}'!")
                    continue
                else:
                    print(f"Traverse graph starting at node '{node_id}' ...")
                    
                # Build query based on filters
                query_parts = ["MATCH (n) WHERE n.id = $node_id"]
                params = {"node_id": node_id}
                
                #for key, value in filters.items():
                #    query_parts.append(f"AND n.{key} = ${key}")
                #    params[key] = value
                
                # Add relevance scoring logic
                if False:
                    query_parts.extend([
                    # PageRank-style scoring
                    "CALL gds.pageRank.stream('graph')",
                    "YIELD nodeId, score as pageRank",
                    "WHERE id(n) = nodeId",
                    
                    # Relationship count scoring
                    "WITH n, pageRank",
                    "MATCH (n)-[r]->()",
                    "WITH n, pageRank, count(r) as outDegree",
                    "MATCH ()-[r]->(n)",
                    "WITH n, pageRank, outDegree, count(r) as inDegree",
                    
                    # Temporal scoring if timestamp exists
                    "OPTIONAL MATCH (n)",
                    "WHERE n.timestamp IS NOT NULL",
                    "WITH n, pageRank, outDegree, inDegree,",
                    "CASE",
                    "  WHEN n.timestamp IS NOT NULL",
                    "  THEN 1 - abs(timestamp() - datetime(n.timestamp).epochMillis) / (365 * 24 * 60 * 60 * 1000.0)",
                    "  ELSE 0.5",
                    "END as temporalScore",
                    
                    # Combine scores
                    "RETURN",
                    "pageRank * 0.4 +",
                    "(outDegree + inDegree) * 0.4 / (CASE WHEN outDegree + inDegree > 0 THEN outDegree + inDegree ELSE 1 END) +",
                    "temporalScore * 0.2 as score"
                    ])
                else:
                    query_parts.extend([
                    "MATCH (n)",
                    "OPTIONAL MATCH (n)-[r1]->()",
                    "WITH n, coalesce(count(r1), 0) AS outDegree",
                    "OPTIONAL MATCH ()-[r2]->(n)",
                    "WITH n, outDegree, coalesce(count(r2), 0) AS inDegree",

                    "WITH n, outDegree, inDegree,",
                    "CASE",
                    "WHEN n.timestamp IS NOT NULL",
                    "THEN 1 - abs(timestamp() - datetime(n.timestamp).epochMillis) / (365 * 24 * 60 * 60 * 1000.0)",
                    "ELSE 0.5",
                    "END AS temporalScore",

                    "RETURN",
                    "(outDegree + inDegree) * 0.6 / (CASE WHEN outDegree + inDegree > 0 THEN outDegree + inDegree ELSE 1 END) + temporalScore * 0.4 AS score"
                    ])
                
                result = session.run(" ".join(query_parts), params)
                record = result.single()
                
                if record:
                    scores[i] = record["score"]
        
        return scores

    def _get_path_to_node(self, node_id: str) -> Optional[List[Dict]]:
        """Find shortest path to node from relevant entry points.
        
        Args:
            node_id: Target node ID
            
        Returns:
            List of nodes and relationships in path, or None if no path found
        """
        with self.neo4j_driver.session() as session:
            # Find shortest path from any entry point
            result = session.run(
                """
                MATCH (start)
                WHERE start.isEntryPoint = true
                MATCH path = shortestPath((start)-[*]->(end))
                WHERE end.id = $node_id
                RETURN path
                ORDER BY length(path)
                LIMIT 1
                """,
                node_id=node_id
            )
            
            record = result.single()
            if not record:
                return None
                
            # Convert path to list of dictionaries
            path = record["path"]
            path_elements = []
            
            for element in path:
                if hasattr(element, "start"):  # Relationship
                    path_elements.append({
                        "type": "relationship",
                        "label": element.type,
                        "properties": dict(element)
                    })
                else:  # Node
                    path_elements.append({
                        "type": "node",
                        "labels": list(element.labels),
                        "properties": dict(element)
                    })
            
            return path_elements

    def close(self):
        """Close database connections."""
        self.neo4j_driver.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
