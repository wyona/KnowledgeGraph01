"""LLM-based implementation of entity and relationship extraction."""

import json
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
import requests
import numpy as np

from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from .base import BaseExtractor
from . import ExtractionResult, ENTITY_TYPES, RELATIONSHIP_SCHEMA

# Pydantic models for structured output parsing
class Entity(BaseModel):
    """Entity extracted from text."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str = Field(description="Type of entity (PERSON, EVENT, etc.)")
    name: str = Field(description="Name or primary identifier of the entity")
    timestamp: Optional[str] = Field(None, description="Timestamp for events")
    properties: Dict = Field(default_factory=dict, description="Additional properties")
    confidence: float = Field(description="Confidence score (0-1)")

class Relationship(BaseModel):
    """Relationship between two entities."""
    type: str = Field(description="Type of relationship")
    subject: Entity = Field(description="Source entity")
    object: Entity = Field(description="Target entity")
    properties: Dict = Field(default_factory=dict, description="Additional properties")
    confidence: float = Field(description="Confidence score (0-1)")

class ExtractionOutput(BaseModel):
    """Complete extraction output."""
    entities: List[Entity]
    relationships: List[Relationship]
    confidence: float

# Extraction prompt template
EXTRACTION_PROMPT = """Extract entities and relationships from the following text. Focus on people, organizations, events, and concepts.

Text: {text}

Instructions:
1. Identify key entities (PERSON, EVENT, ORGANIZATION, CONCEPT, DATE, LOCATION)
2. Extract relationships between entities
3. Include temporal information where relevant
4. Assign confidence scores based on clarity/certainty

Valid relationship types: {valid_relationships}

Output should be structured JSON with the following schema:
{{
    "entities": [
        {{
            "id": "unique_id",
            "type": "PERSON|EVENT|ORGANIZATION|CONCEPT|DATE|LOCATION",
            "name": "entity_name",
            "timestamp": "optional_iso_date",
            "properties": {{}},
            "confidence": 0.0-1.0
        }}
    ],
    "relationships": [
        {{
            "type": "relationship_type",
            "subject": {{entity_object}},
            "object": {{entity_object}},
            "properties": {{}},
            "confidence": 0.0-1.0
        }}
    ],
    "confidence": 0.0-1.0
}}

Text Analysis:
"""

class LLMExtractor(BaseExtractor):
    """Entity and relationship extractor using Ollama."""
    
    def __init__(
        self,
        #model_name: str = "mistral",
        #model_name: str = "llama2",
        model_name: str = "deepseek-r1",
        base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        embedding_dim: int = 768
    ):
        """Initialize the LLM extractor.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Ollama API base URL
            embedding_model: Model to use for embeddings
            embedding_dim: Embedding dimension
        """
        super().__init__()
        
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        
        # Create prompt template
        self.prompt = PromptTemplate(
            input_variables=["text"],
            partial_variables={
                "valid_relationships": ", ".join(RELATIONSHIP_SCHEMA.keys())
            },
            template=EXTRACTION_PROMPT
        )

    def _generate_completion(self, prompt: str) -> str:
        """Generate completion using Ollama API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Model completion
        """
        print(f"Generate completion using Ollama API (Model: {self.model_name}) ...")
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 2000
                }
            }
        )
        response.raise_for_status()
        return response.json()["response"]

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using Ollama API.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        print(f"Get embedding for text '{text}' using Ollama API (Model: {self.embedding_model}) ...")
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.embedding_model,
                "prompt": text
            }
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)

    def get_relevant_entities(self, prompt: str) -> List[Entity]:
        print(f"Get relevant entities using prompt '{prompt}' ...")

        if True:
            response = self._generate_completion(prompt)
            print(f"Response: {response}")

        return None

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relationships from text using LLM.
        
        Args:
            text: Input text to process
            
        Returns:
            ExtractionResult containing extracted information
        """
        # Generate LLM prompt
        prompt = self.prompt.format(text=text)
        #print(f"\nPrompt:\n{prompt}")
        
        # Get structured output from LLM
        if False:
            print(f"\nExtract entities and relationships from text using a LLM:\n{text}\n")
            llm_output = self._generate_completion(prompt)
        else:
            file_mock_entities_relationships = "mock-data/entities-relationship-1.json"
            #file_mock_entities_relationships = "mock-data/entities-relationship-uzh.json"
            print(f"Use mock entities and relationships '{file_mock_entities_relationships}' ...")
            with open(file_mock_entities_relationships, 'r') as file:
                llm_output = file.read()
        
        try:
            # Parse JSON output
            parsed_output = json.loads(llm_output)
            
            # Convert to internal format
            entities = []
            for entity in parsed_output["entities"]:
                entity_dict = {
                    "id": entity["id"],
                    "type": entity["type"],
                    "name": entity["name"],
                    "confidence": entity["confidence"]
                }
                print(f"Extracted entity: {entity_dict}")
                
                # Add optional properties
                if "timestamp" in entity:
                    entity_dict["timestamp"] = entity["timestamp"]
                if "properties" in entity:
                    entity_dict.update(entity["properties"])
                    
                # Add embedding to entity
                if False:
                    print(f"Get embedding for '{entity['name']}' and add to entity ...")
                    entity_dict["embedding"] = self.get_embedding(entity["name"]).tolist()

                entities.append(entity_dict)
                
            relationships = []
            for rel in parsed_output["relationships"]:
                print(f"Extracted relationship: {rel}")
                rel_dict = {
                    "type": rel["type"],
                    "subject": {
                        "id": rel["subject"]["id"],
                        "type": rel["subject"]["type"],
                        "name": rel["subject"]["name"]
                    },
                    "object": {
                        "id": rel["object"]["id"],
                        "type": rel["object"]["type"],
                        "name": rel["object"]["name"]
                    },
                    "confidence": rel["confidence"]
                }
                
                if "properties" in rel:
                    rel_dict.update(rel["properties"])
                    
                relationships.append(rel_dict)
                
            result = ExtractionResult(
                entities=entities,
                relationships=relationships,
                confidence=parsed_output["confidence"]
            )
            
            # Clean and validate results
            return self.clean_extraction(result)
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty result
            return ExtractionResult(
                entities=[],
                relationships=[],
                confidence=0.0
            )

    def batch_extract(self, texts: List[str]) -> ExtractionResult:
        """Process multiple texts and merge results.
        
        Args:
            texts: List of text strings to process
            
        Returns:
            Combined ExtractionResult
        """
        results = [self.extract(text) for text in texts]
        return self.merge_results(results)
