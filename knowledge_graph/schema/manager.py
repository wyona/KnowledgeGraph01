"""Schema manager for Neo4j database operations."""

from typing import List, Optional
from neo4j import GraphDatabase, Driver
from . import (
    PERSON, EVENT, ORGANIZATION, CONCEPT,
    PARTICIPATES_IN, INFLUENCES, RELATED_TO, PART_OF, OCCURS_AT,
    TIMESTAMP, CONFIDENCE, SOURCE, EMBEDDING
)

class SchemaManager:
    """Manages Neo4j database schema, constraints, and indexes."""

    def __init__(self, uri: str, username: str, password: str):
        """Initialize the schema manager with Neo4j connection details.

        Args:
            uri: Neo4j database URI
            username: Database username
            password: Database password
        """
        self.driver: Driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        """Close the database connection."""
        self.driver.close()

    def initialize_schema(self):
        """Set up the initial database schema with constraints and indexes."""
        with self.driver.session() as session:
            # Create constraints for node uniqueness
            constraints = [
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                for label in [PERSON, EVENT, ORGANIZATION, CONCEPT]
            ]
            
            # Create indexes for common property lookups
            indexes = [
                f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{TIMESTAMP})"
                for label in [EVENT]
            ] + [
                f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.name)"
                for label in [PERSON, ORGANIZATION, CONCEPT]
            ]

            # Execute all schema operations
            for query in constraints + indexes:
                session.run(query)

    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_node(self, label: str, properties: dict) -> str:
        """Create a new node with the given label and properties.
        
        Args:
            label: Node label (e.g., PERSON, EVENT)
            properties: Node properties including 'id' and optional properties
        
        Returns:
            The ID of the created node
        """
        with self.driver.session() as session:
            result = session.run(
                f"CREATE (n:{label} $props) RETURN n.id as id",
                props=properties
            )
            return result.single()["id"]

    def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[dict] = None
    ):
        """Create a relationship between two nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            rel_type: Relationship type (e.g., PARTICIPATES_IN)
            properties: Optional relationship properties
        """
        properties = properties or {}
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (a), (b)
                WHERE a.id = $from_id AND b.id = $to_id
                CREATE (a)-[r:{rel_type} $props]->(b)
                RETURN r
                """,
                from_id=from_id,
                to_id=to_id,
                props=properties
            )

    def get_node_by_id(self, node_id: str) -> Optional[dict]:
        """Retrieve a node by its ID.
        
        Args:
            node_id: The unique ID of the node
        
        Returns:
            Node properties as a dictionary, or None if not found
        """
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n {id: $id}) RETURN n",
                id=node_id
            )
            record = result.single()
            return dict(record["n"]) if record else None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
