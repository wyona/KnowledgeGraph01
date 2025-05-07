"""LLM-based query parsing and constraint extraction."""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import requests

from langchain.prompts import PromptTemplate

from . import (
    TimeConstraint,
    EntityConstraint,
    RelationshipConstraint,
    QueryConstraints,
    QUERY_TYPES,
    QUERY_MODIFIERS
)

# Query parsing prompt template
QUERY_PROMPT = """Parse the following natural language query into structured constraints for knowledge graph search.

Query: {query}

Available query types: {query_types}
Available modifiers: {modifiers}

Instructions:
1. Identify the primary query type
2. Extract entity and relationship constraints
3. Parse any temporal constraints
4. Identify query modifiers (limit, paths, etc.)

Output should be structured JSON with the following schema:
{{
    "query_type": "ENTITY_SEARCH|RELATIONSHIP_SEARCH|PATH_SEARCH|TEMPORAL_SEARCH|CAUSAL_SEARCH|SIMILARITY_SEARCH",
    "entities": [
        {{
            "type": "PERSON|EVENT|ORGANIZATION|CONCEPT|DATE|LOCATION",
            "properties": {{}}
        }}
    ],
    "relationships": [
        {{
            "type": "relationship_type",
            "direction": "outgoing|incoming",
            "properties": {{}}
        }}
    ],
    "time": {{
        "start": "optional_iso_date",
        "end": "optional_iso_date",
        "point": "optional_iso_date"
    }},
    "limit": 10,
    "include_paths": false
}}

Query Analysis:
"""

class QueryParser:
    """LLM-based parser for natural language queries."""
    
    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        """Initialize the query parser.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Ollama API base URL
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in model response
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Create prompt template
        self.prompt = PromptTemplate(
            input_variables=["query"],
            partial_variables={
                "query_types": ", ".join(QUERY_TYPES),
                "modifiers": json.dumps(QUERY_MODIFIERS, indent=2)
            },
            template=QUERY_PROMPT
        )

    def _generate_completion(self, prompt: str) -> str:
        """Generate completion using Ollama API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Model completion
        """
        #print(f"Generate completion using Ollama API: {prompt}")
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
        )
        response.raise_for_status()
        return response.json()["response"]

    def parse_query(self, query: str) -> QueryConstraints:
        """
        Parse natural language query into structured constraints.
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryConstraints object containing extracted constraints
        """
        # Generate LLM prompt
        prompt = self.prompt.format(query=query)
        
        # Get structured output from LLM
        if False:
            print(f"Parse query using LLM, in order to get query constraints ...")
            llm_output = self._generate_completion(prompt)
        else:
            file_mock_query_constraints = "mock-data/query-constraints-0.json"
            print(f"Use mock query constraints '{file_mock_query_constraints}' instead using LLM ...")
            with open(file_mock_query_constraints, 'r') as file:
                llm_output = file.read()
        
        try:
            # Parse JSON output
            parsed = json.loads(llm_output)
            
            # Convert temporal constraints
            time_constraint = None
            if parsed.get("time"):
                time_constraint = TimeConstraint(
                    start=datetime.fromisoformat(parsed["time"]["start"]) if parsed["time"].get("start") else None,
                    end=datetime.fromisoformat(parsed["time"]["end"]) if parsed["time"].get("end") else None,
                    point=datetime.fromisoformat(parsed["time"]["point"]) if parsed["time"].get("point") else None
                )
            
            # Convert entity constraints
            entity_constraints = [
                EntityConstraint(
                    type=e["type"],
                    properties=e["properties"]
                )
                for e in parsed.get("entities", [])
            ]
            
            # Convert relationship constraints
            relationship_constraints = [
                RelationshipConstraint(
                    type=r["type"],
                    direction=r["direction"],
                    properties=r["properties"]
                )
                for r in parsed.get("relationships", [])
            ]
            
            return QueryConstraints(
                entities=entity_constraints,
                relationships=relationship_constraints,
                time=time_constraint,
                limit=parsed.get("limit", 10),
                include_paths=parsed.get("include_paths", False)
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            # Return default constraints if parsing fails
            return QueryConstraints(
                entities=[],
                relationships=[],
                time=None,
                limit=10,
                include_paths=False
            )

    def generate_cypher(self, constraints: QueryConstraints) -> Tuple[str, Dict]:
        """Generate Cypher query from constraints.
        
        Args:
            constraints: QueryConstraints object
            
        Returns:
            Tuple of (query_string, parameters)
        """
        # Start with MATCH clauses for entities
        query_parts = []
        params = {}
        
        for i, entity in enumerate(constraints.entities):
            var_name = f"n{i}"
            query_parts.append(f"MATCH ({var_name}:{entity.type})")
            
            # Add property constraints
            if entity.properties:
                conditions = []
                for key, value in entity.properties.items():
                    param_name = f"prop_{i}_{key}"
                    conditions.append(f"{var_name}.{key} = ${param_name}")
                    params[param_name] = value
                    
                if conditions:
                    query_parts.append("WHERE " + " AND ".join(conditions))
        
        # Add relationship patterns
        for i, rel in enumerate(constraints.relationships):
            start_var = f"n{i}"
            end_var = f"n{i+1}"
            
            if rel.direction == "outgoing":
                pattern = f"({start_var})-[r{i}:{rel.type}]->({end_var})"
            else:
                pattern = f"({start_var})<-[r{i}:{rel.type}]-({end_var})"
                
            query_parts.append(f"MATCH {pattern}")
            
            # Add relationship property constraints
            if rel.properties:
                conditions = []
                for key, value in rel.properties.items():
                    param_name = f"rel_{i}_{key}"
                    conditions.append(f"r{i}.{key} = ${param_name}")
                    params[param_name] = value
                    
                if conditions:
                    query_parts.append("WHERE " + " AND ".join(conditions))
        
        # Add temporal constraints if present
        if constraints.time:
            time_conditions = []
            
            if constraints.time.start:
                time_conditions.append("n0.timestamp >= $start_time")
                params["start_time"] = constraints.time.start.isoformat()
                
            if constraints.time.end:
                time_conditions.append("n0.timestamp <= $end_time")
                params["end_time"] = constraints.time.end.isoformat()
                
            if constraints.time.point:
                time_conditions.append("n0.timestamp = $point_time")
                params["point_time"] = constraints.time.point.isoformat()
                
            if time_conditions:
                query_parts.append("WHERE " + " AND ".join(time_conditions))
        
        # Add RETURN clause
        return_items = [f"n{i}" for i in range(len(constraints.entities))]
        if constraints.relationships:
            return_items.extend(f"r{i}" for i in range(len(constraints.relationships)))
            
        query_parts.append("RETURN " + ", ".join(return_items))
        
        # Add LIMIT
        query_parts.append(f"LIMIT {constraints.limit}")
        
        return "\n".join(query_parts), params
