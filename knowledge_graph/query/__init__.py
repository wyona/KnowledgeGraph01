"""Query parsing and execution modules.

This module handles:
- Natural language query understanding
- Query decomposition into search constraints
- Cypher query generation
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TimeConstraint:
    """Temporal constraint extracted from query."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    point: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "point": self.point.isoformat() if self.point else None
        }

@dataclass
class EntityConstraint:
    """Entity-related constraint from query."""
    type: str
    properties: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "type": self.type,
            "properties": self.properties
        }

@dataclass
class RelationshipConstraint:
    """Relationship constraint from query."""
    type: str
    direction: str  # "outgoing" or "incoming"
    properties: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "type": self.type,
            "direction": self.direction,
            "properties": self.properties
        }

@dataclass
class QueryConstraints:
    """Complete set of constraints extracted from query."""
    entities: List[EntityConstraint]
    relationships: List[RelationshipConstraint]
    time: Optional[TimeConstraint] = None
    limit: int = 10
    include_paths: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "time": self.time.to_dict() if self.time else None,
            "limit": self.limit,
            "include_paths": self.include_paths
        }

class QueryResult:
    """Container for query execution results."""
    
    def __init__(
        self,
        nodes: List[Dict],
        relationships: List[Dict],
        paths: Optional[List[List[Dict]]] = None,
        query_time_ms: float = 0
    ):
        """Initialize query result.
        
        Args:
            nodes: List of matched nodes with properties
            relationships: List of matched relationships
            paths: Optional list of paths between nodes
            query_time_ms: Query execution time in milliseconds
        """
        self.nodes = nodes
        self.relationships = relationships
        self.paths = paths
        self.query_time_ms = query_time_ms
        
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "nodes": self.nodes,
            "relationships": self.relationships,
            "paths": self.paths,
            "query_time_ms": self.query_time_ms
        }

# Query types that can be handled
QUERY_TYPES = [
    "ENTITY_SEARCH",      # Find entities matching criteria
    "RELATIONSHIP_SEARCH",  # Find relationships between entities
    "PATH_SEARCH",        # Find paths between entities
    "TEMPORAL_SEARCH",    # Find events/changes over time
    "CAUSAL_SEARCH",      # Find cause-effect relationships
    "SIMILARITY_SEARCH"   # Find similar entities
]

# Query modifiers that affect behavior
QUERY_MODIFIERS = {
    "LIMIT": {
        "type": "integer",
        "description": "Maximum number of results"
    },
    "TIME_WINDOW": {
        "type": "temporal",
        "description": "Time range to search within"
    },
    "INCLUDE_PATHS": {
        "type": "boolean",
        "description": "Whether to return connecting paths"
    },
    "MIN_CONFIDENCE": {
        "type": "float",
        "description": "Minimum confidence threshold"
    }
}

# Default values for query parameters
DEFAULT_QUERY_PARAMS = {
    "limit": 10,
    "min_confidence": 0.5,
    "include_paths": False,
    "vector_weight": 0.5
}
