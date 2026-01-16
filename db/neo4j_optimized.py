"""
BIBLOS v2 - Neo4j Graph Database Client (Optimized)

Implements SPIDERWEB schema for biblical cross-reference graph operations
with comprehensive query optimizations.

Optimization Changes:
1. Connection pooling configuration
2. Batch node/relationship creation using UNWIND
3. Query result limits and pagination
4. Index hints for query optimization
5. Efficient path queries with depth limits
6. Transaction batching for large operations
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from itertools import islice
import logging
import os

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    from neo4j.exceptions import ServiceUnavailable, TransientError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


logger = logging.getLogger("biblos.db.neo4j")


def chunked(iterable, size: int):
    """Yield successive chunks from iterable."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


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


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "biblos2024"
    max_connection_pool_size: int = 50
    max_transaction_retry_time: int = 30
    connection_acquisition_timeout: int = 30
    encrypted: bool = False


class Neo4jClient:
    """
    Optimized Neo4j client for SPIDERWEB graph operations.

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
        password: Optional[str] = None,
        config: Optional[Neo4jConfig] = None
    ):
        self.config = config or Neo4jConfig(
            uri=uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=user or os.getenv("NEO4J_USER", "neo4j"),
            password=password or os.getenv("NEO4J_PASSWORD", "biblos2024")
        )
        self._driver: Optional['AsyncDriver'] = None

    async def connect(self) -> None:
        """Establish connection to Neo4j with optimized pool settings."""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available")
            return

        self._driver = AsyncGraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password),
            # Optimization: Connection pool configuration
            max_connection_pool_size=self.config.max_connection_pool_size,
            max_transaction_retry_time=self.config.max_transaction_retry_time,
            connection_acquisition_timeout=self.config.connection_acquisition_timeout,
            encrypted=self.config.encrypted
        )
        logger.info(
            f"Connected to Neo4j at {self.config.uri} "
            f"(pool_size={self.config.max_connection_pool_size})"
        )

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
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {e}")
            return False

    # =========================================================================
    # Schema initialization
    # =========================================================================

    async def create_indexes(self) -> None:
        """Create indexes and constraints for SPIDERWEB schema."""
        if not self._driver:
            return

        async with self._driver.session() as session:
            # Verse indexes
            await session.run("""
                CREATE INDEX verse_ref IF NOT EXISTS
                FOR (v:Verse) ON (v.reference)
            """)
            await session.run("""
                CREATE INDEX verse_book IF NOT EXISTS
                FOR (v:Verse) ON (v.book)
            """)
            await session.run("""
                CREATE INDEX verse_testament IF NOT EXISTS
                FOR (v:Verse) ON (v.testament)
            """)

            # ChurchFather indexes
            await session.run("""
                CREATE INDEX father_name IF NOT EXISTS
                FOR (f:ChurchFather) ON (f.name)
            """)
            await session.run("""
                CREATE INDEX father_century IF NOT EXISTS
                FOR (f:ChurchFather) ON (f.century)
            """)

            # ThematicCategory indexes
            await session.run("""
                CREATE INDEX theme_name IF NOT EXISTS
                FOR (t:ThematicCategory) ON (t.name)
            """)

            # LexicalEntry indexes
            await session.run("""
                CREATE INDEX lexical_lemma IF NOT EXISTS
                FOR (l:LexicalEntry) ON (l.lemma)
            """)

            # Uniqueness constraints
            await session.run("""
                CREATE CONSTRAINT verse_unique IF NOT EXISTS
                FOR (v:Verse) REQUIRE v.reference IS UNIQUE
            """)
            await session.run("""
                CREATE CONSTRAINT father_unique IF NOT EXISTS
                FOR (f:ChurchFather) REQUIRE f.name IS UNIQUE
            """)
            await session.run("""
                CREATE CONSTRAINT theme_unique IF NOT EXISTS
                FOR (t:ThematicCategory) REQUIRE t.name IS UNIQUE
            """)

            logger.info("Neo4j indexes and constraints created")

    # =========================================================================
    # Verse operations - optimized with batching
    # =========================================================================

    async def create_verse_node(
        self,
        reference: str,
        properties: Dict[str, Any]
    ) -> Optional[str]:
        """Create or update a Verse node with index hint."""
        if not self._driver:
            return None

        async with self._driver.session() as session:
            # Optimization: Use index hint
            result = await session.run("""
                MERGE (v:Verse {reference: $reference})
                USING INDEX v:Verse(reference)
                SET v += $props
                RETURN elementId(v) as id
            """, reference=reference, props=properties)

            record = await result.single()
            return record["id"] if record else None

    async def batch_create_verse_nodes(
        self,
        verses: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        """
        Batch create verse nodes using UNWIND.

        Much more efficient than individual MERGE operations.
        """
        if not self._driver:
            return 0

        count = 0
        async with self._driver.session() as session:
            for batch in chunked(verses, batch_size):
                # Prepare batch data
                batch_data = [
                    {
                        "reference": v["reference"],
                        "properties": {k: v for k, v in v.items() if k != "reference"}
                    }
                    for v in batch
                ]

                result = await session.run("""
                    UNWIND $batch as item
                    MERGE (v:Verse {reference: item.reference})
                    SET v += item.properties
                    RETURN count(v) as created
                """, batch=batch_data)

                record = await result.single()
                count += record["created"] if record else 0

        logger.info(f"Batch created {count} verse nodes")
        return count

    async def get_verse_node(self, reference: str) -> Optional[GraphNode]:
        """Get a Verse node by reference with index hint."""
        if not self._driver:
            return None

        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (v:Verse {reference: $reference})
                USING INDEX v:Verse(reference)
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

    # =========================================================================
    # Cross-reference operations - optimized
    # =========================================================================

    async def create_cross_reference(
        self,
        source_ref: str,
        target_ref: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create a cross-reference relationship with index hints."""
        if not self._driver:
            return None

        if rel_type not in self.RELATIONSHIP_TYPES:
            logger.warning(f"Unknown relationship type: {rel_type}")

        props = properties or {}

        async with self._driver.session() as session:
            # Optimization: Use index hints for both nodes
            query = f"""
                MATCH (source:Verse {{reference: $source_ref}})
                USING INDEX source:Verse(reference)
                MATCH (target:Verse {{reference: $target_ref}})
                USING INDEX target:Verse(reference)
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

    async def batch_create_cross_references(
        self,
        relationships: List[Dict[str, Any]],
        batch_size: int = 500
    ) -> int:
        """
        Batch create cross-reference relationships.

        Each item should have: source_ref, target_ref, rel_type, properties
        """
        if not self._driver:
            return 0

        count = 0
        async with self._driver.session() as session:
            for batch in chunked(relationships, batch_size):
                # Group by relationship type for efficient processing
                by_type: Dict[str, List] = {}
                for rel in batch:
                    rel_type = rel.get("rel_type", "REFERENCES")
                    if rel_type not in by_type:
                        by_type[rel_type] = []
                    by_type[rel_type].append({
                        "source": rel["source_ref"],
                        "target": rel["target_ref"],
                        "props": rel.get("properties", {})
                    })

                # Create relationships by type
                for rel_type, rels in by_type.items():
                    result = await session.run(f"""
                        UNWIND $rels as rel
                        MATCH (source:Verse {{reference: rel.source}})
                        MATCH (target:Verse {{reference: rel.target}})
                        MERGE (source)-[r:{rel_type}]->(target)
                        SET r += rel.props
                        RETURN count(r) as created
                    """, rels=rels)

                    record = await result.single()
                    count += record["created"] if record else 0

        logger.info(f"Batch created {count} cross-reference relationships")
        return count

    async def get_cross_references(
        self,
        verse_ref: str,
        direction: str = "both",
        rel_types: Optional[List[str]] = None,
        limit: int = 100  # Added limit parameter
    ) -> List[Dict[str, Any]]:
        """Get cross-references for a verse with optimized query."""
        if not self._driver:
            return []

        type_filter = ""
        if rel_types:
            type_filter = ":" + "|".join(rel_types)

        async with self._driver.session() as session:
            if direction == "outgoing":
                query = f"""
                    MATCH (source:Verse {{reference: $ref}})-[r{type_filter}]->(target:Verse)
                    USING INDEX source:Verse(reference)
                    RETURN source.reference as source, target.reference as target,
                           type(r) as rel_type, properties(r) as props
                    ORDER BY r.confidence DESC
                    LIMIT $limit
                """
            elif direction == "incoming":
                query = f"""
                    MATCH (source:Verse)-[r{type_filter}]->(target:Verse {{reference: $ref}})
                    USING INDEX target:Verse(reference)
                    RETURN source.reference as source, target.reference as target,
                           type(r) as rel_type, properties(r) as props
                    ORDER BY r.confidence DESC
                    LIMIT $limit
                """
            else:
                query = f"""
                    MATCH (v:Verse {{reference: $ref}})-[r{type_filter}]-(other:Verse)
                    USING INDEX v:Verse(reference)
                    RETURN v.reference as source, other.reference as target,
                           type(r) as rel_type, properties(r) as props
                    ORDER BY r.confidence DESC
                    LIMIT $limit
                """

            result = await session.run(query, ref=verse_ref, limit=limit)
            records = await result.data()
            return records

    # =========================================================================
    # Patristic connections - optimized
    # =========================================================================

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
                USING INDEX f:ChurchFather(name)
                SET f += $props
                RETURN elementId(f) as id
            """, name=name, props=properties)

            record = await result.single()
            return record["id"] if record else None

    async def batch_create_church_fathers(
        self,
        fathers: List[Dict[str, Any]]
    ) -> int:
        """Batch create church father nodes."""
        if not self._driver:
            return 0

        async with self._driver.session() as session:
            result = await session.run("""
                UNWIND $fathers as father
                MERGE (f:ChurchFather {name: father.name})
                SET f += father.properties
                RETURN count(f) as created
            """, fathers=[
                {"name": f["name"], "properties": {k: v for k, v in f.items() if k != "name"}}
                for f in fathers
            ])

            record = await result.single()
            return record["created"] if record else 0

    async def link_father_to_verse(
        self,
        father_name: str,
        verse_ref: str,
        citation_type: str = "CITED_BY",
        properties: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Link a Church Father citation to a verse with index hints."""
        if not self._driver:
            return None

        props = properties or {}

        async with self._driver.session() as session:
            result = await session.run(f"""
                MATCH (f:ChurchFather {{name: $father}})
                USING INDEX f:ChurchFather(name)
                MATCH (v:Verse {{reference: $ref}})
                USING INDEX v:Verse(reference)
                MERGE (f)-[r:{citation_type}]->(v)
                SET r += $props
                RETURN elementId(r) as id
            """, father=father_name, ref=verse_ref, props=props)

            record = await result.single()
            return record["id"] if record else None

    # =========================================================================
    # Thematic connections
    # =========================================================================

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
                USING INDEX t:ThematicCategory(name)
                SET t.description = $description
                RETURN elementId(t) as id
            """, name=name, description=description)

            record = await result.single()
            theme_id = record["id"] if record else None

            # Link to parent if specified
            if theme_id and parent:
                await session.run("""
                    MATCH (child:ThematicCategory {name: $child})
                    USING INDEX child:ThematicCategory(name)
                    MATCH (parent:ThematicCategory {name: $parent})
                    USING INDEX parent:ThematicCategory(name)
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
                USING INDEX v:Verse(reference)
                MATCH (t:ThematicCategory {name: $theme})
                USING INDEX t:ThematicCategory(name)
                MERGE (v)-[r:HAS_THEME]->(t)
                RETURN elementId(r) as id
            """, ref=verse_ref, theme=theme_name)

            record = await result.single()
            return record["id"] if record else None

    # =========================================================================
    # Graph analysis - optimized with limits
    # =========================================================================

    async def find_shortest_path(
        self,
        source_ref: str,
        target_ref: str,
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """Find shortest path between two verses with optimized query."""
        if not self._driver:
            return []

        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (source:Verse {reference: $source})
                USING INDEX source:Verse(reference)
                MATCH (target:Verse {reference: $target})
                USING INDEX target:Verse(reference)
                MATCH path = shortestPath((source)-[*1..$depth]-(target))
                RETURN
                    [n IN nodes(path) | n.reference] as nodes,
                    [r IN relationships(path) | type(r)] as relationships,
                    length(path) as path_length
            """, source=source_ref, target=target_ref, depth=max_depth)

            record = await result.single()
            if record:
                return [{
                    "nodes": record["nodes"],
                    "relationships": record["relationships"],
                    "path_length": record["path_length"]
                }]
            return []

    async def get_verse_neighborhood(
        self,
        verse_ref: str,
        depth: int = 2,
        max_nodes: int = 100,
        max_rels: int = 200
    ) -> Dict[str, Any]:
        """
        Get the neighborhood graph around a verse with limits.

        Optimization: Added LIMIT clauses to prevent massive subgraph returns.
        """
        if not self._driver:
            return {"nodes": [], "relationships": []}

        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (center:Verse {reference: $ref})
                USING INDEX center:Verse(reference)
                CALL {
                    WITH center
                    MATCH path = (center)-[*1..$depth]-(connected)
                    WHERE connected:Verse OR connected:ThematicCategory OR connected:ChurchFather
                    RETURN connected, relationships(path) as rels
                    LIMIT $max_nodes
                }
                WITH collect(DISTINCT connected) as nodes,
                     collect(rels) as all_rels
                RETURN
                    [n IN nodes[..$max_nodes] | {
                        labels: labels(n),
                        properties: properties(n)
                    }] as nodes,
                    [r IN all_rels[..$max_rels] | {
                        type: type(r),
                        properties: properties(r)
                    }] as relationships
            """, ref=verse_ref, depth=depth, max_nodes=max_nodes, max_rels=max_rels)

            record = await result.single()
            if record:
                return {
                    "nodes": record["nodes"],
                    "relationships": record["relationships"]
                }
            return {"nodes": [], "relationships": []}

    async def find_typological_connections(
        self,
        source_ref: str,
        max_depth: int = 3,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find typological connections from OT to NT.

        Optimized Cypher query for common typological analysis.
        """
        if not self._driver:
            return []

        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (source:Verse {reference: $sourceRef})
                USING INDEX source:Verse(reference)
                MATCH path = (source)-[:TYPOLOGICALLY_FULFILLS|PROPHETICALLY_FULFILLS*1..$depth]->(target:Verse)
                WHERE target.testament = 'NT'
                OPTIONAL MATCH (f:ChurchFather)-[:CITED_BY]->(source)
                OPTIONAL MATCH (f2:ChurchFather)-[:CITED_BY]->(target)
                WITH path, target,
                     collect(DISTINCT f.name) as source_fathers,
                     collect(DISTINCT f2.name) as target_fathers
                RETURN
                    target.reference as targetRef,
                    length(path) as distance,
                    [r IN relationships(path) | type(r)] as connectionTypes,
                    source_fathers,
                    target_fathers,
                    size(source_fathers) + size(target_fathers) as patristic_support
                ORDER BY patristic_support DESC, distance
                LIMIT $limit
            """, sourceRef=source_ref, depth=max_depth, limit=limit)

            return await result.data()

    # =========================================================================
    # Statistics - optimized with parallel counting
    # =========================================================================

    async def get_graph_statistics(self) -> Dict[str, int]:
        """Get graph statistics using efficient parallel counting."""
        if not self._driver:
            return {}

        async with self._driver.session() as session:
            # Optimization: Use CALL subqueries for parallel counting
            result = await session.run("""
                CALL {
                    MATCH (v:Verse) RETURN count(v) as verses
                }
                CALL {
                    MATCH (f:ChurchFather) RETURN count(f) as fathers
                }
                CALL {
                    MATCH (t:ThematicCategory) RETURN count(t) as themes
                }
                CALL {
                    MATCH ()-[r]->() RETURN count(r) as relationships
                }
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

    async def get_relationship_statistics(self) -> Dict[str, int]:
        """Get counts by relationship type."""
        if not self._driver:
            return {}

        async with self._driver.session() as session:
            result = await session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """)

            stats = {}
            async for record in result:
                stats[record["rel_type"]] = record["count"]
            return stats
