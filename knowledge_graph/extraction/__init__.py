"""Entity and relationship extraction from unstructured text.

This module provides functionality for extracting entities, relationships,
and temporal information from text using LLMs and structured parsing.
"""

from typing import Dict, List, Optional, Tuple

# Entity types that can be extracted
ENTITY_TYPES = [
    "PERSON",
    "EVENT",
    "ORGANIZATION",
    "UNIVERSITY",
    "CONCEPT",
    "DATE",
    "LOCATION"
]

# Relationship extraction schema
RELATIONSHIP_SCHEMA = {
    "PARTICIPATES_IN": {
        "valid_subjects": ["PERSON", "ORGANIZATION"],
        "valid_objects": ["EVENT"]
    },
    "INFLUENCES": {
        "valid_subjects": ["PERSON", "ORGANIZATION", "EVENT", "CONCEPT"],
        "valid_objects": ["PERSON", "ORGANIZATION", "EVENT", "CONCEPT"]
    },
    "FATHER_OF": {
        "valid_subjects": ["PERSON"],
        "valid_objects": ["PERSON"]
    },
    "STUDIES_AT": {
        "valid_subjects": ["PERSON"],
        "valid_objects": ["UNIVERSITY"]
    },
    "HUSBAND_OF": {
        "valid_subjects": ["PERSON"],
        "valid_objects": ["PERSON"]
    },
    "BROTHER_OF": {
        "valid_subjects": ["PERSON"],
        "valid_objects": ["PERSON"]
    },
    "RELATED_TO": {
        "valid_subjects": ["PERSON", "ORGANIZATION", "EVENT", "CONCEPT"],
        "valid_objects": ["PERSON", "ORGANIZATION", "EVENT", "CONCEPT"]
    },
    "PART_OF": {
        "valid_subjects": ["PERSON", "ORGANIZATION", "EVENT", "CONCEPT"],
        "valid_objects": ["ORGANIZATION", "EVENT", "CONCEPT"]
    },
    "OCCURS_AT": {
        "valid_subjects": ["EVENT"],
        "valid_objects": ["DATE", "LOCATION"]
    }
}

# Required properties for each entity type
ENTITY_PROPERTIES = {
    "PERSON": ["name", "id"],
    "EVENT": ["name", "id", "timestamp"],
    "ORGANIZATION": ["name", "id"],
    "UNIVERSITY": ["name", "id"],
    "CONCEPT": ["name", "id"],
    "DATE": ["value", "id"],
    "LOCATION": ["name", "id"]
}

class ExtractionResult:
    """Container for extraction results including entities and relationships."""
    
    def __init__(
        self,
        entities: List[Dict],
        relationships: List[Dict],
        confidence: float
    ):
        """Initialize extraction result.
        
        Args:
            entities: List of extracted entities with their properties
            relationships: List of extracted relationships between entities
            confidence: Confidence score for the extraction (0-1)
        """
        self.entities = entities
        self.relationships = relationships
        self.confidence = confidence

    def to_dict(self) -> Dict:
        """Convert extraction result to dictionary format."""
        return {
            "entities": self.entities,
            "relationships": self.relationships,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExtractionResult":
        """Create ExtractionResult from dictionary data."""
        return cls(
            entities=data["entities"],
            relationships=data["relationships"],
            confidence=data["confidence"]
        )
