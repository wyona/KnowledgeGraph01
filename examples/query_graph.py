"""Example script demonstrating natural language querying of the knowledge graph."""

import os
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from dotenv import load_dotenv

# Add parent directory to path to import knowledge_graph package
sys.path.append(str(Path(__file__).parent.parent))

from knowledge_graph.query.parser import QueryParser
from knowledge_graph.search.hybrid import HybridSearchEngine
from knowledge_graph.search.vector_store import VectorStore
from knowledge_graph.extraction.llm import LLMExtractor

# Load environment variables
load_dotenv()

def format_result(result: Dict) -> str:
    """Format a search result for display.
    
    Args:
        result: Search result dictionary
        
    Returns:
        Formatted string representation
    """
    output = []
    
    # Add main entity information
    output.append(f"Entity: {result['properties'].get('name', 'Unknown')}")
    output.append(f"Type: {result['properties'].get('type', 'Unknown')}")
    output.append(f"Confidence: {result['score']:.2f}")
    
    # Add temporal information if available
    if "timestamp" in result["properties"]:
        output.append(f"Time: {result['properties']['timestamp']}")
    
    # Add path information if available
    if result.get("path"):
        path = []
        for element in result["path"]:
            if element["type"] == "node":
                path.append(element["properties"].get("name", "Unknown"))
            else:  # relationship
                path.append(f"--[{element['label']}]-->")
        output.append("Path: " + " ".join(path))
    
    return "\n".join(output)

def main():
    """Main query demonstration script."""
    # Initialize components
    vector_store = VectorStore.load(
        os.getenv("VECTOR_STORE_PATH", "./vector_store")
    )
    
    search_engine = HybridSearchEngine(
        vector_store=vector_store,
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD")
    )
    
    query_parser = QueryParser()
    extractor = LLMExtractor()
    
    # Example queries to demonstrate different types of searches
    example_queries_1 = [
        "Who founded Google and when?",
        "What contributions did Alan Turing make during World War II?",
        "How is the Turing Award related to computer science?",
        "What major acquisitions has Google made?",
        "Show me the timeline of Google's major events from 1998 to 2015",
        "Find connections between Turing and artificial intelligence",
    ]
    example_queries = [
        "Was studiert der älteste Sohn von Michael?",
    ]
    
    print("Knowledge Graph Query Examples\n")
    
    for i, query in enumerate(example_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)
        
        try:
            # Parse query into constraints
            print("Parse query using LLM ...")
            constraints = query_parser.parse_query(query)
            print(f"Constraints of parsed query: {constraints}")
            
            # Get query embedding for hybrid search
            print(f"Get embedding for query '{query}' ...")
            query_embedding = extractor.get_embedding(query)
            
            # Execute hybrid search
            print(f"Search vector store using embedding '{query_embedding}'")
            results = search_engine.search(
                query_vector=query_embedding,
                params={
                    "vector_weight": 0.5,
                    "max_results": 5,
                    "include_paths": True,
                    "filters": constraints.to_dict()
                }
            )
            
            # Display results
            print(f"\nFound {results.total_found} results in {results.query_time_ms:.2f}ms:\n")
            
            for j, result in enumerate(results.results, 1):
                print(f"Result {j}:")
                print(format_result(result.to_dict()))
                print()
                
        except Exception as e:
            print(f"Error processing query: {e}")
        
        print("-" * 50)
        
    # Cleanup
    search_engine.close()

if __name__ == "__main__":
    main()
