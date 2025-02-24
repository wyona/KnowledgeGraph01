"""Neo4j schema definitions and database management.

This module defines the core schema for our knowledge graph, including:
- Node types (Person, Event, Organization, Concept)
- Relationship types and their properties
- Constraints and indexes for efficient querying
"""

from typing import Dict, List, Optional, Union

# Node labels
PERSON = "Person"
EVENT = "Event"
ORGANIZATION = "Organization"
CONCEPT = "Concept"

# Relationship types
PARTICIPATES_IN = "PARTICIPATES_IN"
INFLUENCES = "INFLUENCES"
RELATED_TO = "RELATED_TO"
PART_OF = "PART_OF"
OCCURS_AT = "OCCURS_AT"

# Property keys
TIMESTAMP = "timestamp"
CONFIDENCE = "confidence"
SOURCE = "source"
EMBEDDING = "embedding"
