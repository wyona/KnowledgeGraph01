"""Main FastAPI application for the knowledge graph system."""

import os
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from . import (
    ExtractionRequest, ExtractionResponse,
    QueryRequest, QueryResponse,
    SearchRequest, SearchResponse,
    SystemStatus, HealthCheck, ErrorResponse
)
from ..schema.manager import SchemaManager
from ..extraction.llm import LLMExtractor
from ..search.vector_store import VectorStore
from ..search.hybrid import HybridSearchEngine
from ..query.parser import QueryParser

# Load environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # nomic-embed-text dimension

# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Graph API",
    description="API for knowledge graph construction and querying",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
schema_manager = SchemaManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
extractor = LLMExtractor(
    model_name=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    embedding_model=OLLAMA_EMBEDDING_MODEL,
    embedding_dim=EMBEDDING_DIM
)
vector_store = VectorStore(dimension=EMBEDDING_DIM)
search_engine = HybridSearchEngine(
    vector_store=vector_store,
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD
)
query_parser = QueryParser(
    model_name=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL
)

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    # Initialize Neo4j schema
    schema_manager.initialize_schema()
    
    # Load vector store if exists
    if os.path.exists(VECTOR_STORE_PATH):
        global vector_store
        vector_store = VectorStore.load(VECTOR_STORE_PATH)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    # Save vector store
    vector_store.save(VECTOR_STORE_PATH)
    
    # Close connections
    schema_manager.close()
    search_engine.close()

@app.post("/extract", response_model=ExtractionResponse)
async def extract_from_text(request: ExtractionRequest):
    """Extract entities and relationships from text."""
    try:
        start_time = datetime.now()
        
        # Extract entities and relationships
        result = extractor.extract(request.text)
        
        # Filter by confidence threshold
        entities = [
            e for e in result.entities 
            if e.get("confidence", 0) >= request.confidence_threshold
        ]
        
        relationships = []
        if request.extract_relationships:
            relationships = [
                r for r in result.relationships
                if r.get("confidence", 0) >= request.confidence_threshold
            ]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ExtractionResponse(
            entities=entities,
            relationships=relationships,
            confidence=result.confidence,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Extraction failed", "message": str(e)}
        )

@app.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """Execute natural language query."""
    try:
        start_time = datetime.now()
        
        # Parse query into constraints
        constraints = query_parser.parse_query(request.query)
        
        # Update constraints with request parameters
        constraints.limit = request.max_results
        constraints.include_paths = request.include_paths
        
        # Generate and execute Cypher query
        cypher_query, params = query_parser.generate_cypher(constraints)
        
        with schema_manager.driver.session() as session:
            result = session.run(cypher_query, params)
            records = [dict(record) for record in result]
        
        query_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResponse(
            results=records,
            total_found=len(records),
            query_time_ms=query_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Query execution failed", "message": str(e)}
        )

@app.post("/search", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """Perform hybrid vector + graph search."""
    try:
        # Generate query embedding
        query_embedding = extractor.get_embedding(request.text)
        
        # Set up search parameters
        search_params = {
            "vector_weight": request.vector_weight,
            "max_results": request.max_results,
            "include_paths": request.include_paths
        }
        
        # Add filters
        if request.entity_types:
            search_params["filters"] = {"type": request.entity_types}
            
        if request.time_range:
            search_params["filters"].update(request.time_range)
        
        # Execute search
        results = search_engine.search(
            query_vector=query_embedding,
            params=search_params
        )
        
        return SearchResponse(
            results=[r.to_dict() for r in results.results],
            total_found=results.total_found,
            query_time_ms=results.query_time_ms
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Search failed", "message": str(e)}
        )

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status."""
    try:
        with schema_manager.driver.session() as session:
            # Get Neo4j stats
            result = session.run("""
                MATCH (n)
                OPTIONAL MATCH ()-[r]->()
                RETURN count(distinct n) as nodes, count(distinct r) as rels
            """)
            stats = result.single()
            
        return SystemStatus(
            neo4j_connected=True,
            total_nodes=stats["nodes"],
            total_relationships=stats["rels"],
            vector_store_size=vector_store.total_vectors,
            last_updated=vector_store.last_updated.isoformat() 
                if vector_store.last_updated else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Status check failed", "message": str(e)}
        )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check system health status."""
    components = {
        "neo4j": True,
        "vector_store": True,
        "extractor": True,
        "ollama": True
    }
    
    try:
        # Check Neo4j connection
        schema_manager.driver.verify_connectivity()
    except:
        components["neo4j"] = False
    
    try:
        # Check vector store
        vector_store.index.ntotal
    except:
        components["vector_store"] = False
    
    try:
        # Check Ollama connection
        extractor.get_embedding("test")
    except:
        components["ollama"] = False
        components["extractor"] = False
    
    status = "healthy" if all(components.values()) else "degraded"
    
    return HealthCheck(
        status=status,
        version="0.1.0",
        components=components
    )
