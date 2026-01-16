"""
BIBLOS v2 - Neo4j Graph Database Client

Implements SPIDERWEB schema for biblical cross-reference graph operations.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import os

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    from neo4j.exceptions import (
        ServiceUnavailable,
        AuthError,
        SessionExpired,
        TransientError,
        DatabaseError as Neo4jDatabaseError,
    )
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    ServiceUnavailable = Exception
    AuthError = Exception
    SessionExpired = Exception
    TransientError = Exception
    Neo4jDatabaseError = Exception

# Import core error types for specific exception handling
from core.errors import (
    BiblosError,
    BiblosDatabaseError,
    BiblosResourceError,
)

logger = logging.getLogger("biblos.db.neo4j")


@dataclass
class GraphNode:
    """Represents a node in the graph."""
    id: str
    labels: List[str]
    properties: Dict[str, Any]


@dataclass
class GraphRelationship:
    """Represents a relationship in the graph."""
    id: str
    type: str
    start_node: str
    end_node: str
    properties: Dict[str, Any]


class Neo4jClient:
    """
    Neo4j client for SPIDERWEB graph operations.

    Node Types:
    - Verse: Biblical verses
    - ChurchFather: Patristic authors
    - ThematicCategory: 31 thematic categories
    - LiturgicalContext: Liturgical usage contexts
    - LexicalEntry: Greek/Hebrew vocabulary

    Relationship Types:
    - REFERENCES: Direct scripture reference
    - QUOTES: Verbal quotation
    - ALLUDES_TO: Allusion
    - TYPOLOGICALLY_FULFILLS: Type/antitype
    - PROPHETICALLY_FULFILLS: Prophecy fulfillment
    - THEMATICALLY_CONNECTED: Thematic link
    - LITURGICALLY_USED: Liturgical connection
    """

    RELATIONSHIP_TYPES = [
        "REFERENCES", "QUOTES", "ALLUDES_TO", "TYPOLOGICALLY_FULFILLS",
        "PROPHETICALLY_FULFILLS", "THEMATICALLY_CONNECTED", "LITURGICALLY_USED",
        "VERBAL_PARALLEL", "NARRATIVE_PARALLEL", "CITED_BY"
    ]

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "biblos2024")
        self._driver: Optional['AsyncDriver'] = None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available")
            return

        self._driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )
        logger.info(f"Connected to Neo4j at {self.uri}")

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j connection closed")

    async def verify_connectivity(self) -> bool:
        """Verify database connectivity."""
        if not self._driver:
            return False
        try:
            await self._driver.verify_connectivity()
            return True
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            return False
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            return False
        except SessionExpired as e:
            logger.warning(f"Neo4j session expired: {e}")
            return False
        except TransientError as e:
            logger.warning(f"Neo4j transient error: {e}")
            return False
        except Neo4jDatabaseError as e:
            logger.error(f"Neo4j database error: {e}")
            return False
        except (MemoryError, BiblosResourceError) as e:
            logger.critical(f"Resource exhaustion during Neo4j connectivity check: {e}")
            raise
        except BiblosDatabaseError as e:
            logger.error(f"Database error during connectivity check: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Neo4j connectivity check: {e} ({type(e).__name__})")
            return False

    # Schema initialization
    async def create_indexes(self) -> None:
        """Create indexes and constraints for SPIDERWEB schema."""
        if not self._driver:
            return

        async with self._driver.session() as session:
            # Verse index
            await session.run("""
                CREATE INDEX verse_ref IF NOT EXISTS
                FOR (v:Verse) ON (v.reference)
            """)

            # ChurchFather index
            await session.run("""
                CREATE INDEX father_name IF NOT EXISTS
                FOR (f:ChurchFather) ON (f.name)
            """)

            # ThematicCategory index
            await session.run("""
                CREATE INDEX theme_name IF NOT EXISTS
                FOR (t:ThematicCategory) ON (t.name)
            """)

            # Uniqueness constraint
            await session.run("""
                CREATE CONSTRAINT verse_unique IF NOT EXISTS
                FOR (v:Verse) REQUIRE v.reference IS UNIQUE
            """)

            logger.info("Neo4j indexes created")

    # Verse operations
    async def create_verse_node(
        self,
        reference: str,
        properties: Dict[str, Any]
    ) -> Optional[str]:
        """Create or update a Verse node."""
        if not self._driver:
            return None

        async with self._driver.session() as session:
            result = await session.run("""
                MERGE (v:Verse {reference: $reference})
                SET v += $props
                RETURN elementId(v) as id
            """, reference=reference, props=properties)

            record = await result.single()
            return record["id"] if record else None

    async def get_verse_node(self, reference: str) -> Optional[GraphNode]:
        """Get a Verse node by reference."""
        if not self._driver:
            return None

        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (v:Verse {reference: $reference})
                RETURN v, labels(v) as labels, elementId(v) as id
            """, reference=reference)

            record = await result.single()
            if record:
                return GraphNode(
                    id=record["id"],
                    labels=record["labels"],
                    properties=dict(record["v"])
                )
            return None

    # Cross-reference operations
    async def create_cross_reference(
        self,
        source_ref: str,
        target_ref: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create a cross-reference relationship."""
        if not self._driver:
            return None

        if rel_type not in self.RELATIONSHIP_TYPES:
            logger.warning(f"Unknown relationship type: {rel_type}")

        props = properties or {}

        async with self._driver.session() as session:
            query = f"""
                MATCH (source:Verse {{reference: $source_ref}})
                MATCH (target:Verse {{reference: $target_ref}})
                MERGE (source)-[r:{rel_type}]->(target)
                SET r += $props
                RETURN elementId(r) as id
            """
            result = await session.run(
                query,
                source_ref=source_ref,
                target_ref=target_ref,
                props=props
            )

            record = await result.single()
            return record["id"] if record else None

    async def get_cross_references(
        self,
        verse_ref: str,
        direction: str = "both",
        rel_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get cross-references for a verse."""
        if not self._driver:
            return []

        type_filter = ""
        if rel_types:
            type_filter = ":" + "|".join(rel_types)

        async with self._driver.session() as session:
            if direction == "outgoing":
                query = f"""
                    MATCH (source:Verse {{reference: $ref}})-[r{type_filter}]->(target:Verse)
                    RETURN source.reference as source, target.reference as target,
                           type(r) as rel_type, properties(r) as props
                """
            elif direction == "incoming":
                query = f"""
                    MATCH (source:Verse)-[r{type_filter}]->(target:Verse {{reference: $ref}})
                    RETURN source.reference as source, target.reference as target,
                           type(r) as rel_type, properties(r) as props
                """
            else:
                query = f"""
                    MATCH (v:Verse {{reference: $ref}})-[r{type_filter}]-(other:Verse)
                    RETURN v.reference as source, other.reference as target,
                           type(r) as rel_type, properties(r) as props
                """

            result = await session.run(query, ref=verse_ref)
            records = await result.data()
            return records

    # Patristic connections
    async def create_church_father(
        self,
        name: str,
        properties: Dict[str, Any]
    ) -> Optional[str]:
        """Create a ChurchFather node."""
        if not self._driver:
            return None

        async with self._driver.session() as session:
            result = await session.run("""
                MERGE (f:ChurchFather {name: $name})
                SET f += $props
                RETURN elementId(f) as id
            """, name=name, props=properties)

            record = await result.single()
            return record["id"] if record else None

    async def link_father_to_verse(
        self,
        father_name: str,
        verse_ref: str,
        citation_type: str = "CITED_BY",
        properties: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Link a Church Father citation to a verse."""
        if not self._driver:
            return None

        props = properties or {}

        async with self._driver.session() as session:
            result = await session.run(f"""
                MATCH (f:ChurchFather {{name: $father}})
                MATCH (v:Verse {{reference: $ref}})
                MERGE (f)-[r:{citation_type}]->(v)
                SET r += $props
                RETURN elementId(r) as id
            """, father=father_name, ref=verse_ref, props=props)

            record = await result.single()
            return record["id"] if record else None

    # Thematic connections
    async def create_thematic_category(
        self,
        name: str,
        description: str,
        parent: Optional[str] = None
    ) -> Optional[str]:
        """Create a ThematicCategory node."""
        if not self._driver:
            return None

        async with self._driver.session() as session:
            result = await session.run("""
                MERGE (t:ThematicCategory {name: $name})
                SET t.description = $description
                RETURN elementId(t) as id
            """, name=name, description=description)

            record = await result.single()
            theme_id = record["id"] if record else None

            # Link to parent if specified
            if theme_id and parent:
                await session.run("""
                    MATCH (child:ThematicCategory {name: $child})
                    MATCH (parent:ThematicCategory {name: $parent})
                    MERGE (child)-[:SUBCATEGORY_OF]->(parent)
                """, child=name, parent=parent)

            return theme_id

    async def tag_verse_with_theme(
        self,
        verse_ref: str,
        theme_name: str
    ) -> Optional[str]:
        """Tag a verse with a thematic category."""
        if not self._driver:
            return None

        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (v:Verse {reference: $ref})
                MATCH (t:ThematicCategory {name: $theme})
                MERGE (v)-[r:HAS_THEME]->(t)
                RETURN elementId(r) as id
            """, ref=verse_ref, theme=theme_name)

            record = await result.single()
            return record["id"] if record else None

    # Graph analysis
    async def find_shortest_path(
        self,
        source_ref: str,
        target_ref: str,
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """Find shortest path between two verses."""
        if not self._driver:
            return []

        async with self._driver.session() as session:
            result = await session.run("""
                MATCH path = shortestPath(
                    (source:Verse {reference: $source})-[*1..$depth]-(target:Verse {reference: $target})
                )
                RETURN [n IN nodes(path) | n.reference] as nodes,
                       [r IN relationships(path) | type(r)] as relationships
            """, source=source_ref, target=target_ref, depth=max_depth)

            record = await result.single()
            if record:
                return [{
                    "nodes": record["nodes"],
                    "relationships": record["relationships"]
                }]
            return []

    async def get_verse_neighborhood(
        self,
        verse_ref: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """Get the neighborhood graph around a verse."""
        if not self._driver:
            return {"nodes": [], "relationships": []}

        async with self._driver.session() as session:
            result = await session.run("""
                MATCH path = (center:Verse {reference: $ref})-[*1..$depth]-(connected)
                WITH collect(DISTINCT connected) as nodes,
                     collect(DISTINCT relationships(path)) as rels
                RETURN nodes, rels
            """, ref=verse_ref, depth=depth)

            record = await result.single()
            if record:
                return {
                    "nodes": [dict(n) for n in record["nodes"]],
                    "relationships": record["rels"]
                }
            return {"nodes": [], "relationships": []}

    # Statistics
    async def get_graph_statistics(self) -> Dict[str, int]:
        """Get graph statistics."""
        if not self._driver:
            return {}

        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (v:Verse) WITH count(v) as verses
                MATCH (f:ChurchFather) WITH verses, count(f) as fathers
                MATCH (t:ThematicCategory) WITH verses, fathers, count(t) as themes
                MATCH ()-[r]->() WITH verses, fathers, themes, count(r) as relationships
                RETURN verses, fathers, themes, relationships
            """)

            record = await result.single()
            if record:
                return {
                    "verses": record["verses"],
                    "church_fathers": record["fathers"],
                    "thematic_categories": record["themes"],
                    "relationships": record["relationships"]
                }
            return {}
