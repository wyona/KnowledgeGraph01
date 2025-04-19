"""Example script demonstrating data ingestion into the knowledge graph."""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from dotenv import load_dotenv

# Add parent directory to path to import knowledge_graph package
sys.path.append(str(Path(__file__).parent.parent))

from knowledge_graph.schema.manager import SchemaManager
from knowledge_graph.extraction.llm import LLMExtractor
from knowledge_graph.search.vector_store import VectorStore

# Load environment variables
load_dotenv()

def process_text_file(
    filepath: str,
    extractor: LLMExtractor,
    schema_manager: SchemaManager,
    vector_store: VectorStore,
    batch_size: int = 10
) -> Dict:
    """Process a text file and extract knowledge into the graph.
    
    Args:
        filepath: Path to text file
        extractor: Initialized LLMExtractor
        schema_manager: Initialized SchemaManager
        vector_store: Initialized VectorStore
        batch_size: Number of paragraphs to process at once
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        "total_entities": 0,
        "total_relationships": 0,
        "processing_time_ms": 0
    }
    
    start_time = datetime.now()
    
    # Read and split text into paragraphs
    with open(filepath, 'r') as f:
        text = f.read()
        
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Process in batches
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        
        # Extract entities and relationships
        for text in batch:
            print(f"\nExtract entities and relationships from text:\n{text}")
            result = extractor.extract(text)
            #print(f"\nCompletion result:\n{result.to_dict()}")
            
            # Store entities
            for entity in result.entities:
                try:
                    # Create node in Neo4j
                    response = input(f"\nDo you want to add entity '{entity['name']}' to Knowledge Graph / Neo4j? (YES/no): ").strip().lower()
                    if response in ['', 'yes', 'y']:
                        print(f"\tAdd entity to Neo4j: {entity['name']}")
                        node_id = schema_manager.create_node(
                            label=entity["type"],
                            properties=entity
                        )

                        # Store embedding in vector store
                        if "embedding" in entity:
                            #print(f"Embedding:\n{entity['embedding']}")
                            print(f"\tAdd embedding of entity '{node_id}' to vector store ...")
                            vector_store.add_vectors(
                               vectors=np.array([entity["embedding"]], dtype=np.float32),
                               node_ids=[node_id]
                            )

                        stats["total_entities"] += 1
                    else:
                        print("\tEntity skipped.")
                    
                except Exception as e:
                    print(f"Error storing entity: {e}")
            
            # Store relationships
            for rel in result.relationships:
                try:
                    response = input(f"\nDo you want to add relationship '{rel}' to Knowledge Graph / Neo4j? (YES/no): ").strip().lower()
                    if response in ['', 'yes', 'y']:
                        print(f"\tAdd relationship to Neo4j: {rel}")
                        schema_manager.create_relationship(
                            from_id=rel["subject"]["id"],
                            to_id=rel["object"]["id"],
                            rel_type=rel["type"],
                            properties={
                                k: v for k, v in rel.items()
                                if k not in ["subject", "object", "type"]
                            }
                        )
                    
                        stats["total_relationships"] += 1
                    else:
                        print("\tRelationship skipped.")
                    
                except Exception as e:
                    print(f"Error storing relationship: {e}")
    
    stats["processing_time_ms"] = (
        datetime.now() - start_time
    ).total_seconds() * 1000
    
    return stats

def main():
    """Main ingestion script."""
    # Initialize components
    schema_manager = SchemaManager(
        uri=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    extractor = LLMExtractor()
    
    vector_store = VectorStore(
        dimension=int(os.getenv("EMBEDDING_DIM", 1536))
    )
    
    # Process sample data
    sample_files = [
        "data/sample1.txt",
        "data/sample2.txt"
    ]
    
    total_stats = {
        "total_entities": 0,
        "total_relationships": 0,
        "total_time_ms": 0
    }
    
    for filepath in sample_files:
        print(f"\nProcessing {filepath}...")
        
        try:
            stats = process_text_file(
                filepath=filepath,
                extractor=extractor,
                schema_manager=schema_manager,
                vector_store=vector_store,
                batch_size=int(os.getenv("BATCH_SIZE", 10))
            )
            
            print(f"Extracted {stats['total_entities']} entities")
            print(f"Created {stats['total_relationships']} relationships")
            print(f"Processing time: {stats['processing_time_ms']:.2f}ms")
            
            # Update total stats
            total_stats["total_entities"] += stats["total_entities"]
            total_stats["total_relationships"] += stats["total_relationships"]
            total_stats["total_time_ms"] += stats["processing_time_ms"]
            
        except Exception as e:
            print(f"Error processing file: {e}")
    
    print("\nIngestion complete!")
    print(f"Total entities: {total_stats['total_entities']}")
    print(f"Total relationships: {total_stats['total_relationships']}")
    print(f"Total processing time: {total_stats['total_time_ms']:.2f}ms")
    
    # Save vector store
    vector_store.save(os.getenv("VECTOR_STORE_PATH", "./vector_store"))
    
    # Cleanup
    schema_manager.close()

if __name__ == "__main__":
    main()
