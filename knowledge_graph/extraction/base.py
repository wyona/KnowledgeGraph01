"""Base class for entity and relationship extractors."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from . import ExtractionResult, ENTITY_TYPES, RELATIONSHIP_SCHEMA, ENTITY_PROPERTIES

class BaseExtractor(ABC):
    """Abstract base class for text extraction implementations."""

    def __init__(self):
        """Initialize the extractor with schema validation helpers."""
        self.entity_types = ENTITY_TYPES
        self.relationship_schema = RELATIONSHIP_SCHEMA
        self.entity_properties = ENTITY_PROPERTIES

    @abstractmethod
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relationships from text.
        
        Args:
            text: Input text to process
            
        Returns:
            ExtractionResult containing extracted entities and relationships
        """
        pass

    def validate_entity(self, entity: Dict) -> bool:
        """Validate an extracted entity against the schema.
        
        Args:
            entity: Dictionary containing entity properties
            
        Returns:
            True if entity is valid, False otherwise
        """
        if "type" not in entity or entity["type"] not in self.entity_types:
            return False

        required_props = self.entity_properties[entity["type"]]
        return all(prop in entity for prop in required_props)

    def validate_relationship(self, relationship: Dict) -> bool:
        """Validate an extracted relationship against the schema.
        
        Args:
            relationship: Dictionary containing relationship properties
            
        Returns:
            True if relationship is valid, False otherwise
        """
        if "type" not in relationship or relationship["type"] not in self.relationship_schema:
            return False

        schema = self.relationship_schema[relationship["type"]]
        
        # Check if subject and object entities exist and have valid types
        if "subject" not in relationship or "object" not in relationship:
            return False
            
        subject = relationship["subject"]
        object_ = relationship["object"]
        
        if not (self.validate_entity(subject) and self.validate_entity(object_)):
            return False
            
        # Validate entity type combinations
        if (subject["type"] not in schema["valid_subjects"] or
            object_["type"] not in schema["valid_objects"]):
            return False
            
        return True

    def clean_extraction(self, result: ExtractionResult) -> ExtractionResult:
        """Clean and validate extraction results.
        
        Args:
            result: Raw extraction result
            
        Returns:
            Cleaned ExtractionResult with only valid entities and relationships
        """
        # Filter valid entities
        valid_entities = [
            entity for entity in result.entities
            if self.validate_entity(entity)
        ]
        
        # Filter valid relationships
        valid_relationships = [
            rel for rel in result.relationships
            if self.validate_relationship(rel)
        ]
        
        # Update confidence based on validation ratio
        if result.entities:
            entity_ratio = len(valid_entities) / len(result.entities)
            rel_ratio = (len(valid_relationships) / len(result.relationships) 
                        if result.relationships else 1.0)
            new_confidence = result.confidence * (entity_ratio + rel_ratio) / 2
        else:
            new_confidence = 0.0
            
        return ExtractionResult(
            entities=valid_entities,
            relationships=valid_relationships,
            confidence=new_confidence
        )

    def merge_results(self, results: List[ExtractionResult]) -> ExtractionResult:
        """Merge multiple extraction results, removing duplicates.
        
        Args:
            results: List of ExtractionResults to merge
            
        Returns:
            Combined ExtractionResult with deduplicated entities and relationships
        """
        merged_entities = {}
        merged_relationships = []
        total_confidence = 0.0
        
        # Merge entities, keeping highest confidence version of duplicates
        for result in results:
            for entity in result.entities:
                entity_id = entity["id"]
                if (entity_id not in merged_entities or 
                    merged_entities[entity_id].get("confidence", 0) < entity.get("confidence", 0)):
                    merged_entities[entity_id] = entity
            
            # Add all relationships (duplicates handled in clean_extraction)
            merged_relationships.extend(result.relationships)
            total_confidence += result.confidence
            
        avg_confidence = total_confidence / len(results) if results else 0.0
        
        result = ExtractionResult(
            entities=list(merged_entities.values()),
            relationships=merged_relationships,
            confidence=avg_confidence
        )
        
        return self.clean_extraction(result)
