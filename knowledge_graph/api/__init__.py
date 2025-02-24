"""FastAPI service for the knowledge graph system.

This module provides REST API endpoints for:
- Entity and relationship extraction
- Query parsing and execution
- Graph search and traversal
"""

from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# API models
class ExtractionRequest(BaseModel):
    """Request model for text extraction."""
    text: str
    extract_relationships: bool = True
    confidence_threshold: float = 0.5

class ExtractionResponse(BaseModel):
    """Response model for extraction results."""
    entities: List[Dict]
    relationships: List[Dict]
    confidence: float
    processing_time_ms: float

class QueryRequest(BaseModel):
    """Request model for knowledge graph queries."""
    query: str
    max_results: int = 10
    include_paths: bool = False
    min_confidence: float = 0.5
    vector_weight: float = 0.5

class QueryResponse(BaseModel):
    """Response model for query results."""
    results: List[Dict]
    total_found: int
    query_time_ms: float

class SearchRequest(BaseModel):
    """Request model for hybrid search."""
    text: str
    entity_types: Optional[List[str]] = None
    time_range: Optional[Dict[str, str]] = None
    max_results: int = 10
    include_paths: bool = False
    vector_weight: float = 0.5

class SearchResponse(BaseModel):
    """Response model for search results."""
    results: List[Dict]
    total_found: int
    query_time_ms: float

# Error responses
class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    details: Optional[Dict] = None

# API status/health models
class SystemStatus(BaseModel):
    """System status information."""
    neo4j_connected: bool
    total_nodes: int
    total_relationships: int
    vector_store_size: int
    last_updated: str

class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: Dict[str, bool]
