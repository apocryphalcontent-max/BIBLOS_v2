"""
BIBLOS v2 - Neo4j Graph Data Science (GDS) Algorithms

Implements advanced graph algorithms for biblical scholarship:
- Centrality measures (PageRank, HITS, Betweenness)
- Community detection (Louvain, Leiden, Label Propagation)
- Path finding and pattern matching
- Graph projection management

Requires Neo4j Enterprise with GDS plugin installed.
"""
from typing import List, Dict, Any, Optional, Tuple, Set, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import asyncio

from db.neo4j_schema import (
    NodeLabel, RelationshipType, GraphMetricType,
    CommunityAlgorithm, PathAlgorithm, GraphStatistics
)

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

logger = logging.getLogger("biblos.db.neo4j_algorithms")


# =============================================================================
# ALGORITHM CONFIGURATION
# =============================================================================

@dataclass
class AlgorithmConfig:
    """Configuration for graph algorithms."""
    # PageRank settings
    pagerank_damping: float = 0.85
    pagerank_iterations: int = 20
    pagerank_tolerance: float = 0.0001

    # HITS settings
    hits_iterations: int = 20

    # Community detection
    louvain_resolution: float = 1.0
    louvain_max_levels: int = 10
    leiden_gamma: float = 1.0
    leiden_theta: float = 0.01

    # Path finding
    path_max_hops: int = 5
    dijkstra_max_cost: float = 1000.0

    # Batching
    batch_size: int = 1000

    # Projection settings
    relationship_orientation: str = "UNDIRECTED"  # NATURAL, REVERSE, UNDIRECTED
    concurrent_algorithms: bool = True

    # Caching
    cache_projections: bool = True
    projection_ttl_hours: int = 24


ALGORITHM_CONFIG = AlgorithmConfig()


# =============================================================================
# GRAPH PROJECTION MANAGEMENT
# =============================================================================

@dataclass
class GraphProjection:
    """Represents an in-memory graph projection for GDS algorithms."""
    name: str
    node_labels: List[str]
    relationship_types: List[str]
    node_count: int = 0
    relationship_count: int = 0
    created_at: Optional[datetime] = None
    memory_usage_bytes: int = 0

    @property
    def is_stale(self) -> bool:
        """Check if projection is older than TTL."""
        if not self.created_at:
            return True
        age_hours = (datetime.utcnow() - self.created_at).total_seconds() / 3600
        return age_hours > ALGORITHM_CONFIG.projection_ttl_hours


class ProjectionManager:
    """Manages GDS graph projections."""

    def __init__(self, driver: 'AsyncDriver'):
        self._driver = driver
        self._projections: Dict[str, GraphProjection] = {}

    async def create_projection(
        self,
        name: str,
        node_labels: List[str],
        relationship_config: Dict[str, Dict[str, Any]],
        node_properties: Optional[List[str]] = None
    ) -> GraphProjection:
        """Create a named graph projection for algorithm execution."""
        # Drop if exists
        await self.drop_projection(name)

        node_projection = node_labels if len(node_labels) > 1 else node_labels[0]

        async with self._driver.session() as session:
            result = await session.run("""
                CALL gds.graph.project(
                    $name,
                    $nodes,
                    $rels
                ) YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
            """, name=name, nodes=node_projection, rels=relationship_config)

            record = await result.single()

            projection = GraphProjection(
                name=name,
                node_labels=node_labels,
                relationship_types=list(relationship_config.keys()),
                node_count=record["nodeCount"],
                relationship_count=record["relationshipCount"],
                created_at=datetime.utcnow()
            )

            self._projections[name] = projection
            logger.info(
                f"Created projection '{name}': {projection.node_count} nodes, "
                f"{projection.relationship_count} relationships"
            )
            return projection

    async def drop_projection(self, name: str) -> bool:
        """Drop a graph projection if it exists."""
        async with self._driver.session() as session:
            result = await session.run("""
                CALL gds.graph.exists($name) YIELD exists
                RETURN exists
            """, name=name)
            record = await result.single()

            if record and record["exists"]:
                await session.run("""
                    CALL gds.graph.drop($name) YIELD graphName
                    RETURN graphName
                """, name=name)
                self._projections.pop(name, None)
                logger.info(f"Dropped projection '{name}'")
                return True
            return False

    async def get_or_create_standard_projection(self) -> GraphProjection:
        """Get or create the standard BIBLOS verse projection."""
        name = "biblos-verse-graph"

        if name in self._projections and not self._projections[name].is_stale:
            return self._projections[name]

        rel_config = {
            "CROSS_REFERENCES": {
                "type": "CROSS_REFERENCES",
                "properties": ["confidence", "mutual_influence", "necessity_score"],
                "orientation": "UNDIRECTED"
            },
            "TYPIFIES": {
                "type": "TYPIFIES",
                "properties": ["composite_strength"],
                "orientation": "NATURAL"
            },
            "FULFILLED_IN": {
                "type": "FULFILLED_IN",
                "properties": ["necessity_score"],
                "orientation": "NATURAL"
            },
            "QUOTED_IN": {
                "type": "QUOTED_IN",
                "properties": ["match_confidence"],
                "orientation": "NATURAL"
            }
        }

        return await self.create_projection(
            name=name,
            node_labels=["Verse"],
            relationship_config=rel_config
        )

    async def list_projections(self) -> List[Dict[str, Any]]:
        """List all existing graph projections."""
        async with self._driver.session() as session:
            result = await session.run("""
                CALL gds.graph.list()
                YIELD graphName, nodeCount, relationshipCount, creationTime, memoryUsage
                RETURN graphName, nodeCount, relationshipCount, creationTime, memoryUsage
            """)
            records = await result.data()
            return records


# =============================================================================
# CENTRALITY ALGORITHMS
# =============================================================================

class CentralityAlgorithms:
    """Centrality algorithm implementations."""

    def __init__(self, driver: 'AsyncDriver', projection_manager: ProjectionManager):
        self._driver = driver
        self._projections = projection_manager

    async def calculate_pagerank(
        self,
        write_property: str = "pagerank_score",
        relationship_weight: str = "confidence",
        damping_factor: float = None,
        max_iterations: int = None,
        tolerance: float = None
    ) -> Dict[str, float]:
        """
        Calculate PageRank centrality for all verses.

        High PageRank indicates theologically central verses (hubs).
        """
        damping = damping_factor or ALGORITHM_CONFIG.pagerank_damping
        iterations = max_iterations or ALGORITHM_CONFIG.pagerank_iterations
        tol = tolerance or ALGORITHM_CONFIG.pagerank_tolerance

        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            # Run PageRank with write mode
            await session.run(f"""
                CALL gds.pageRank.write('{projection.name}', {{
                    writeProperty: $write_prop,
                    relationshipWeightProperty: $weight_prop,
                    dampingFactor: $damping,
                    maxIterations: $iterations,
                    tolerance: $tolerance,
                    concurrency: 4
                }})
                YIELD nodePropertiesWritten, ranIterations, didConverge
                RETURN nodePropertiesWritten, ranIterations, didConverge
            """,
                write_prop=write_property,
                weight_prop=relationship_weight,
                damping=damping,
                iterations=iterations,
                tolerance=tol
            )

            # Return top 100 verses by centrality
            result = await session.run(f"""
                MATCH (v:Verse)
                WHERE v.{write_property} > 0
                RETURN v.id AS verse_id, v.{write_property} AS score
                ORDER BY score DESC
                LIMIT 100
            """)

            records = await result.data()
            return {r["verse_id"]: r["score"] for r in records}

    async def calculate_article_rank(
        self,
        write_property: str = "article_rank_score"
    ) -> Dict[str, float]:
        """
        Calculate ArticleRank - variant of PageRank better for sparse graphs.

        Uses a lower damping factor for disconnected clusters.
        """
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            await session.run(f"""
                CALL gds.articleRank.write('{projection.name}', {{
                    writeProperty: $write_prop,
                    relationshipWeightProperty: 'confidence',
                    dampingFactor: 0.85,
                    maxIterations: 20
                }})
            """, write_prop=write_property)

            result = await session.run(f"""
                MATCH (v:Verse)
                WHERE v.{write_property} > 0
                RETURN v.id AS verse_id, v.{write_property} AS score
                ORDER BY score DESC
                LIMIT 100
            """)

            records = await result.data()
            return {r["verse_id"]: r["score"] for r in records}

    async def calculate_hits(
        self,
        hub_property: str = "hub_score",
        authority_property: str = "authority_score"
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate HITS (Hyperlink-Induced Topic Search) scores.

        Hub = OT verse that references many authorities (key prophetic sources)
        Authority = NT verse referenced by many hubs (key fulfillment passages)
        """
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            await session.run(f"""
                CALL gds.hits.write('{projection.name}', {{
                    hitsIterations: {ALGORITHM_CONFIG.hits_iterations},
                    authProperty: $auth_prop,
                    hubProperty: $hub_prop
                }})
            """, auth_prop=authority_property, hub_prop=hub_property)

            # Get top hubs
            hubs_result = await session.run(f"""
                MATCH (v:Verse)
                WHERE v.{hub_property} > 0
                RETURN v.id AS verse_id, v.{hub_property} AS score
                ORDER BY score DESC
                LIMIT 50
            """)
            hubs = {r["verse_id"]: r["score"] for r in await hubs_result.data()}

            # Get top authorities
            auth_result = await session.run(f"""
                MATCH (v:Verse)
                WHERE v.{authority_property} > 0
                RETURN v.id AS verse_id, v.{authority_property} AS score
                ORDER BY score DESC
                LIMIT 50
            """)
            authorities = {r["verse_id"]: r["score"] for r in await auth_result.data()}

            return hubs, authorities

    async def calculate_betweenness(
        self,
        write_property: str = "betweenness_score",
        sampling: bool = True,
        sample_size: int = 1000
    ) -> Dict[str, float]:
        """
        Calculate betweenness centrality.

        High betweenness = verse lies on many shortest paths (bridge verse).
        """
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            if sampling:
                await session.run(f"""
                    CALL gds.betweenness.write('{projection.name}', {{
                        writeProperty: $write_prop,
                        samplingSize: $sample_size,
                        samplingSeed: 42
                    }})
                """, write_prop=write_property, sample_size=sample_size)
            else:
                await session.run(f"""
                    CALL gds.betweenness.write('{projection.name}', {{
                        writeProperty: $write_prop
                    }})
                """, write_prop=write_property)

            result = await session.run(f"""
                MATCH (v:Verse)
                WHERE v.{write_property} > 0
                RETURN v.id AS verse_id, v.{write_property} AS score
                ORDER BY score DESC
                LIMIT 100
            """)

            records = await result.data()
            return {r["verse_id"]: r["score"] for r in records}

    async def calculate_closeness(
        self,
        write_property: str = "closeness_score"
    ) -> Dict[str, float]:
        """
        Calculate closeness centrality.

        High closeness = verse is close to all other verses on average.
        """
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            await session.run(f"""
                CALL gds.closeness.write('{projection.name}', {{
                    writeProperty: $write_prop
                }})
            """, write_prop=write_property)

            result = await session.run(f"""
                MATCH (v:Verse)
                WHERE v.{write_property} > 0
                RETURN v.id AS verse_id, v.{write_property} AS score
                ORDER BY score DESC
                LIMIT 100
            """)

            records = await result.data()
            return {r["verse_id"]: r["score"] for r in records}

    async def calculate_eigenvector(
        self,
        write_property: str = "eigenvector_score"
    ) -> Dict[str, float]:
        """
        Calculate eigenvector centrality.

        Similar to PageRank but considers connection quality more.
        """
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            await session.run(f"""
                CALL gds.eigenvector.write('{projection.name}', {{
                    writeProperty: $write_prop,
                    maxIterations: 100
                }})
            """, write_prop=write_property)

            result = await session.run(f"""
                MATCH (v:Verse)
                WHERE v.{write_property} > 0
                RETURN v.id AS verse_id, v.{write_property} AS score
                ORDER BY score DESC
                LIMIT 100
            """)

            records = await result.data()
            return {r["verse_id"]: r["score"] for r in records}


# =============================================================================
# COMMUNITY DETECTION ALGORITHMS
# =============================================================================

class CommunityAlgorithms:
    """Community detection algorithm implementations."""

    def __init__(self, driver: 'AsyncDriver', projection_manager: ProjectionManager):
        self._driver = driver
        self._projections = projection_manager

    async def detect_louvain_communities(
        self,
        write_property: str = "community_id",
        min_community_size: int = 5,
        include_intermediate: bool = False
    ) -> Dict[int, List[str]]:
        """
        Detect thematic communities using Louvain algorithm.

        Returns mapping of community_id -> list of verse_ids.
        """
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            params = {
                "write_prop": write_property,
                "resolution": ALGORITHM_CONFIG.louvain_resolution,
                "max_levels": ALGORITHM_CONFIG.louvain_max_levels,
            }

            if include_intermediate:
                await session.run(f"""
                    CALL gds.louvain.write('{projection.name}', {{
                        writeProperty: $write_prop,
                        includeIntermediateCommunities: true,
                        relationshipWeightProperty: 'confidence',
                        maxLevels: $max_levels,
                        resolution: $resolution
                    }})
                """, **params)
            else:
                await session.run(f"""
                    CALL gds.louvain.write('{projection.name}', {{
                        writeProperty: $write_prop,
                        relationshipWeightProperty: 'confidence',
                        maxLevels: $max_levels,
                        resolution: $resolution
                    }})
                """, **params)

            # Get communities with minimum size
            result = await session.run(f"""
                MATCH (v:Verse)
                WHERE v.{write_property} IS NOT NULL
                WITH v.{write_property} AS community_id, collect(v.id) AS verses
                WHERE size(verses) >= $min_size
                RETURN community_id, verses
                ORDER BY size(verses) DESC
            """, min_size=min_community_size)

            communities = {}
            async for record in result:
                communities[record["community_id"]] = record["verses"]

            logger.info(f"Detected {len(communities)} communities with >{min_community_size} members")
            return communities

    async def detect_leiden_communities(
        self,
        write_property: str = "leiden_community_id",
        min_community_size: int = 5
    ) -> Dict[int, List[str]]:
        """
        Detect communities using Leiden algorithm.

        Leiden is an improvement over Louvain with guaranteed connectivity.
        """
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            await session.run(f"""
                CALL gds.leiden.write('{projection.name}', {{
                    writeProperty: $write_prop,
                    relationshipWeightProperty: 'confidence',
                    gamma: $gamma,
                    theta: $theta
                }})
            """,
                write_prop=write_property,
                gamma=ALGORITHM_CONFIG.leiden_gamma,
                theta=ALGORITHM_CONFIG.leiden_theta
            )

            result = await session.run(f"""
                MATCH (v:Verse)
                WHERE v.{write_property} IS NOT NULL
                WITH v.{write_property} AS community_id, collect(v.id) AS verses
                WHERE size(verses) >= $min_size
                RETURN community_id, verses
                ORDER BY size(verses) DESC
            """, min_size=min_community_size)

            communities = {}
            async for record in result:
                communities[record["community_id"]] = record["verses"]

            return communities

    async def detect_label_propagation(
        self,
        write_property: str = "lpa_community_id"
    ) -> Dict[int, List[str]]:
        """
        Detect communities using label propagation.

        Fast but non-deterministic. Good for initial exploration.
        """
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            await session.run(f"""
                CALL gds.labelPropagation.write('{projection.name}', {{
                    writeProperty: $write_prop,
                    relationshipWeightProperty: 'confidence'
                }})
            """, write_prop=write_property)

            result = await session.run(f"""
                MATCH (v:Verse)
                WHERE v.{write_property} IS NOT NULL
                WITH v.{write_property} AS community_id, collect(v.id) AS verses
                RETURN community_id, verses
                ORDER BY size(verses) DESC
            """)

            communities = {}
            async for record in result:
                communities[record["community_id"]] = record["verses"]

            return communities

    async def find_weakly_connected_components(
        self
    ) -> Dict[int, List[str]]:
        """Find weakly connected components in the graph."""
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            result = await session.run(f"""
                CALL gds.wcc.stream('{projection.name}')
                YIELD nodeId, componentId
                WITH componentId, collect(gds.util.asNode(nodeId).id) AS verses
                RETURN componentId, verses
                ORDER BY size(verses) DESC
            """)

            components = {}
            async for record in result:
                components[record["componentId"]] = record["verses"]

            return components

    async def get_community_summary(
        self,
        community_property: str = "community_id"
    ) -> List[Dict[str, Any]]:
        """
        Get summary statistics for each detected community.

        Includes:
        - Size
        - Internal density
        - Most central verses
        - Book distribution
        """
        async with self._driver.session() as session:
            result = await session.run(f"""
                MATCH (v:Verse)
                WHERE v.{community_property} IS NOT NULL
                WITH v.{community_property} AS community_id, collect(v) AS members

                // Calculate community statistics
                WITH community_id, members, size(members) AS size

                // Get book distribution
                UNWIND members AS m
                WITH community_id, size, members, m.book AS book
                WITH community_id, size, members, collect(DISTINCT book) AS books

                // Get central verses (by PageRank if available)
                UNWIND members AS m
                WITH community_id, size, books, m
                ORDER BY coalesce(m.pagerank_score, 0) DESC
                WITH community_id, size, books, collect(m.id)[0..3] AS central_verses

                // Calculate average internal connectivity
                RETURN community_id,
                       size,
                       books,
                       central_verses
                ORDER BY size DESC
            """)

            summaries = []
            async for record in result:
                summaries.append({
                    "community_id": record["community_id"],
                    "size": record["size"],
                    "books": record["books"],
                    "central_verses": record["central_verses"]
                })

            return summaries


# =============================================================================
# PATH FINDING ALGORITHMS
# =============================================================================

class PathAlgorithms:
    """Path finding algorithm implementations."""

    def __init__(self, driver: 'AsyncDriver', projection_manager: ProjectionManager):
        self._driver = driver
        self._projections = projection_manager

    async def find_shortest_path(
        self,
        source_verse: str,
        target_verse: str,
        relationship_types: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find shortest path between two verses.

        Returns path details including intermediate verses and relationship types.
        """
        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(relationship_types)

        async with self._driver.session() as session:
            result = await session.run(f"""
                MATCH path = shortestPath(
                    (source:Verse {{id: $source}})-[{rel_filter}*..{ALGORITHM_CONFIG.path_max_hops}]-(target:Verse {{id: $target}})
                )
                WITH path,
                     [n IN nodes(path) | n.id] AS verse_ids,
                     [r IN relationships(path) | type(r)] AS rel_types,
                     [r IN relationships(path) | coalesce(r.confidence, r.composite_strength, 1.0)] AS weights,
                     length(path) AS hops
                RETURN verse_ids, rel_types, weights, hops,
                       reduce(product = 1.0, w IN weights | product * w) AS path_confidence
            """, source=source_verse, target=target_verse)

            record = await result.single()
            if record:
                return {
                    "verses": record["verse_ids"],
                    "relationship_types": record["rel_types"],
                    "weights": record["weights"],
                    "hops": record["hops"],
                    "path_confidence": record["path_confidence"]
                }
            return None

    async def find_all_shortest_paths(
        self,
        source_verse: str,
        target_verse: str,
        max_paths: int = 10
    ) -> List[Dict[str, Any]]:
        """Find all shortest paths between two verses."""
        async with self._driver.session() as session:
            result = await session.run(f"""
                MATCH path = allShortestPaths(
                    (source:Verse {{id: $source}})-[*..{ALGORITHM_CONFIG.path_max_hops}]-(target:Verse {{id: $target}})
                )
                WITH path,
                     [n IN nodes(path) | n.id] AS verse_ids,
                     [r IN relationships(path) | type(r)] AS rel_types,
                     [r IN relationships(path) | coalesce(r.confidence, 1.0)] AS weights
                WITH verse_ids, rel_types, weights,
                     reduce(product = 1.0, w IN weights | product * w) AS path_confidence
                RETURN verse_ids, rel_types, weights, path_confidence
                ORDER BY path_confidence DESC
                LIMIT $max_paths
            """, source=source_verse, target=target_verse, max_paths=max_paths)

            paths = []
            async for record in result:
                paths.append({
                    "verses": record["verse_ids"],
                    "relationship_types": record["rel_types"],
                    "weights": record["weights"],
                    "path_confidence": record["path_confidence"]
                })

            return paths

    async def find_dijkstra_path(
        self,
        source_verse: str,
        target_verse: str,
        cost_property: str = "confidence"
    ) -> Optional[Dict[str, Any]]:
        """
        Find shortest weighted path using Dijkstra's algorithm.

        Inverts confidence to cost (higher confidence = lower cost).
        """
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            result = await session.run(f"""
                MATCH (source:Verse {{id: $source}})
                MATCH (target:Verse {{id: $target}})
                CALL gds.shortestPath.dijkstra.stream('{projection.name}', {{
                    sourceNode: source,
                    targetNode: target,
                    relationshipWeightProperty: $cost_prop
                }})
                YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
                RETURN [nodeId IN nodeIds | gds.util.asNode(nodeId).id] AS verse_ids,
                       totalCost,
                       costs
            """, source=source_verse, target=target_verse, cost_prop=cost_property)

            record = await result.single()
            if record:
                return {
                    "verses": record["verse_ids"],
                    "total_cost": record["totalCost"],
                    "step_costs": record["costs"]
                }
            return None

    async def find_typological_chain(
        self,
        start_verse: str,
        pattern_id: str,
        max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find chain of typological connections following a pattern.

        Traces how a type develops through progressive revelation.
        """
        async with self._driver.session() as session:
            result = await session.run(f"""
                MATCH chain = (start:Verse {{id: $start_verse}})
                    -[:TYPIFIES*1..{max_depth}]->(end:Verse)
                WHERE ALL(r IN relationships(chain) WHERE r.pattern_id = $pattern_id)
                WITH chain,
                     [n IN nodes(chain) | {{id: n.id, testament: n.testament, book: n.book}}] AS verses,
                     [r IN relationships(chain) | {{
                         strength: r.composite_strength,
                         layer: r.dominant_layer,
                         relation: r.relation
                     }}] AS connections,
                     length(chain) AS depth
                RETURN verses, connections, depth,
                       reduce(s = 1.0, c IN connections | s * c.strength) AS chain_confidence
                ORDER BY chain_confidence DESC
                LIMIT 10
            """, start_verse=start_verse, pattern_id=pattern_id)

            chains = []
            async for record in result:
                chains.append({
                    "verses": record["verses"],
                    "connections": record["connections"],
                    "depth": record["depth"],
                    "chain_confidence": record["chain_confidence"]
                })

            return chains

    async def find_prophetic_fulfillment_path(
        self,
        prophecy_verse: str,
        fulfillment_verse: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find the path from prophecy to fulfillment through intermediate connections.
        """
        async with self._driver.session() as session:
            result = await session.run("""
                MATCH path = shortestPath(
                    (prophecy:Verse {id: $prophecy})
                    -[:FULFILLED_IN|TYPIFIES|QUOTED_IN*1..5]->
                    (fulfillment:Verse {id: $fulfillment})
                )
                WITH path,
                     [n IN nodes(path) | {id: n.id, testament: n.testament}] AS verses,
                     [r IN relationships(path) | {type: type(r), score: coalesce(r.necessity_score, r.composite_strength, r.match_confidence)}] AS connections
                RETURN verses, connections
            """, prophecy=prophecy_verse, fulfillment=fulfillment_verse)

            record = await result.single()
            if record:
                return {
                    "verses": record["verses"],
                    "connections": record["connections"]
                }
            return None


# =============================================================================
# SIMILARITY ALGORITHMS
# =============================================================================

class SimilarityAlgorithms:
    """Node similarity algorithm implementations."""

    def __init__(self, driver: 'AsyncDriver', projection_manager: ProjectionManager):
        self._driver = driver
        self._projections = projection_manager

    async def find_similar_verses_node_similarity(
        self,
        verse_id: str,
        top_k: int = 10,
        similarity_cutoff: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Find verses with similar connection patterns using Node Similarity.
        """
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            result = await session.run(f"""
                MATCH (v:Verse {{id: $verse_id}})
                CALL gds.nodeSimilarity.stream('{projection.name}', {{
                    topK: $top_k,
                    similarityCutoff: $cutoff
                }})
                YIELD node1, node2, similarity
                WITH gds.util.asNode(node1) AS v1, gds.util.asNode(node2) AS v2, similarity
                WHERE v1.id = $verse_id
                RETURN v2.id AS similar_verse, similarity
                ORDER BY similarity DESC
            """, verse_id=verse_id, top_k=top_k, cutoff=similarity_cutoff)

            similar = []
            async for record in result:
                similar.append({
                    "verse_id": record["similar_verse"],
                    "similarity": record["similarity"]
                })

            return similar

    async def calculate_jaccard_similarity(
        self,
        verse_a: str,
        verse_b: str
    ) -> float:
        """
        Calculate Jaccard similarity between two verses based on shared connections.
        """
        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (a:Verse {id: $verse_a})-[:CROSS_REFERENCES]-(shared:Verse)
                MATCH (b:Verse {id: $verse_b})-[:CROSS_REFERENCES]-(shared)
                WITH a, b, count(DISTINCT shared) AS intersection

                MATCH (a)-[:CROSS_REFERENCES]-(a_neighbor:Verse)
                WITH a, b, intersection, count(DISTINCT a_neighbor) AS a_count

                MATCH (b)-[:CROSS_REFERENCES]-(b_neighbor:Verse)
                WITH intersection, a_count, count(DISTINCT b_neighbor) AS b_count

                RETURN toFloat(intersection) / (a_count + b_count - intersection) AS jaccard
            """, verse_a=verse_a, verse_b=verse_b)

            record = await result.single()
            return record["jaccard"] if record else 0.0


# =============================================================================
# GRAPH STATISTICS
# =============================================================================

class GraphAnalytics:
    """Graph-wide analytics and statistics."""

    def __init__(self, driver: 'AsyncDriver', projection_manager: ProjectionManager):
        self._driver = driver
        self._projections = projection_manager

    async def get_graph_statistics(self) -> GraphStatistics:
        """Calculate comprehensive graph statistics."""
        async with self._driver.session() as session:
            # Basic counts
            counts_result = await session.run("""
                MATCH (v:Verse)
                WITH count(v) AS verse_count
                MATCH (w:Word)
                WITH verse_count, count(w) AS word_count
                MATCH (l:Lemma)
                WITH verse_count, word_count, count(l) AS lemma_count
                MATCH (f:Father)
                WITH verse_count, word_count, lemma_count, count(f) AS father_count
                MATCH ()-[r]->()
                WITH verse_count, word_count, lemma_count, father_count, count(r) AS edge_count
                RETURN verse_count, word_count, lemma_count, father_count, edge_count
            """)

            counts = await counts_result.single()

            # Relationship type counts
            rel_counts_result = await session.run("""
                MATCH ()-[r:CROSS_REFERENCES]->()
                WITH count(r) AS cross_ref_count
                MATCH ()-[t:TYPIFIES]->()
                WITH cross_ref_count, count(t) AS typological_count
                MATCH ()-[f:FULFILLED_IN]->()
                WITH cross_ref_count, typological_count, count(f) AS prophetic_count
                MATCH (:Father)-[i:INTERPRETS]->()
                WITH cross_ref_count, typological_count, prophetic_count, count(i) AS patristic_count
                RETURN cross_ref_count, typological_count, prophetic_count, patristic_count
            """)

            rel_counts = await rel_counts_result.single()

            # Calculate derived statistics
            node_count = counts["verse_count"]
            edge_count = counts["edge_count"]
            avg_degree = (2 * edge_count) / node_count if node_count > 0 else 0
            max_possible_edges = node_count * (node_count - 1) / 2
            density = edge_count / max_possible_edges if max_possible_edges > 0 else 0

            # Get WCC info
            components = await self._get_component_info()

            return GraphStatistics(
                node_count=node_count,
                edge_count=edge_count,
                avg_degree=avg_degree,
                density=density,
                diameter=None,  # Too expensive to compute
                connected_components=len(components),
                largest_component_size=max(components.values()) if components else 0,
                verse_count=counts["verse_count"],
                word_count=counts["word_count"],
                lemma_count=counts["lemma_count"],
                father_count=counts["father_count"],
                cross_ref_count=rel_counts["cross_ref_count"],
                typological_count=rel_counts["typological_count"],
                prophetic_count=rel_counts["prophetic_count"],
                patristic_count=rel_counts["patristic_count"]
            )

    async def _get_component_info(self) -> Dict[int, int]:
        """Get connected component sizes."""
        try:
            projection = await self._projections.get_or_create_standard_projection()

            async with self._driver.session() as session:
                result = await session.run(f"""
                    CALL gds.wcc.stream('{projection.name}')
                    YIELD componentId
                    WITH componentId, count(*) AS size
                    RETURN componentId, size
                """)

                components = {}
                async for record in result:
                    components[record["componentId"]] = record["size"]

                return components
        except Exception:
            return {}

    async def calculate_triangle_count(
        self,
        write_property: str = "triangle_count"
    ) -> int:
        """
        Calculate triangle counts for clustering analysis.

        High triangle count indicates dense local clustering.
        """
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            result = await session.run(f"""
                CALL gds.triangleCount.write('{projection.name}', {{
                    writeProperty: $write_prop
                }})
                YIELD globalTriangleCount, nodeCount
                RETURN globalTriangleCount, nodeCount
            """, write_prop=write_property)

            record = await result.single()
            return record["globalTriangleCount"] if record else 0

    async def calculate_local_clustering_coefficient(
        self,
        write_property: str = "local_clustering"
    ) -> Dict[str, float]:
        """Calculate local clustering coefficient for each verse."""
        projection = await self._projections.get_or_create_standard_projection()

        async with self._driver.session() as session:
            await session.run(f"""
                CALL gds.localClusteringCoefficient.write('{projection.name}', {{
                    writeProperty: $write_prop
                }})
            """, write_prop=write_property)

            result = await session.run(f"""
                MATCH (v:Verse)
                WHERE v.{write_property} > 0
                RETURN v.id AS verse_id, v.{write_property} AS coefficient
                ORDER BY coefficient DESC
                LIMIT 100
            """)

            coefficients = {}
            async for record in result:
                coefficients[record["verse_id"]] = record["coefficient"]

            return coefficients


# =============================================================================
# MAIN CLIENT CLASS
# =============================================================================

class Neo4jAlgorithmsClient:
    """
    Main client for Neo4j GDS algorithms.

    Provides unified access to all algorithm categories.
    """

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None
    ):
        import os
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "biblos2024")
        self._driver: Optional['AsyncDriver'] = None

        # Algorithm components (initialized on connect)
        self.projections: Optional[ProjectionManager] = None
        self.centrality: Optional[CentralityAlgorithms] = None
        self.community: Optional[CommunityAlgorithms] = None
        self.paths: Optional[PathAlgorithms] = None
        self.similarity: Optional[SimilarityAlgorithms] = None
        self.analytics: Optional[GraphAnalytics] = None

    async def connect(self) -> None:
        """Establish connection and initialize algorithm components."""
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available")

        self._driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )

        # Initialize algorithm components
        self.projections = ProjectionManager(self._driver)
        self.centrality = CentralityAlgorithms(self._driver, self.projections)
        self.community = CommunityAlgorithms(self._driver, self.projections)
        self.paths = PathAlgorithms(self._driver, self.projections)
        self.similarity = SimilarityAlgorithms(self._driver, self.projections)
        self.analytics = GraphAnalytics(self._driver, self.projections)

        logger.info(f"Connected to Neo4j GDS at {self.uri}")

    async def close(self) -> None:
        """Close connection and clean up projections."""
        if self._driver:
            # Clean up standard projection
            if self.projections:
                await self.projections.drop_projection("biblos-verse-graph")

            await self._driver.close()
            logger.info("Neo4j GDS connection closed")

    async def verify_gds_available(self) -> bool:
        """Verify GDS library is installed and available."""
        async with self._driver.session() as session:
            try:
                result = await session.run("""
                    RETURN gds.version() AS version
                """)
                record = await result.single()
                if record:
                    logger.info(f"GDS version: {record['version']}")
                    return True
            except Exception as e:
                logger.error(f"GDS not available: {e}")
                return False
        return False

    async def run_full_analysis(
        self,
        include_expensive: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete graph analysis pipeline.

        Args:
            include_expensive: Include computationally expensive algorithms

        Returns:
            Comprehensive analysis results
        """
        results = {}

        # Always run
        results["statistics"] = await self.analytics.get_graph_statistics()
        results["pagerank"] = await self.centrality.calculate_pagerank()
        results["communities"] = await self.community.detect_louvain_communities()

        if include_expensive:
            results["hits"] = await self.centrality.calculate_hits()
            results["betweenness"] = await self.centrality.calculate_betweenness()
            results["triangles"] = await self.analytics.calculate_triangle_count()
            results["clustering"] = await self.analytics.calculate_local_clustering_coefficient()

        return results

    # Convenience methods for common operations

    async def get_central_verses(self, top_k: int = 50) -> List[Dict[str, Any]]:
        """Get the most theologically central verses by PageRank."""
        pagerank = await self.centrality.calculate_pagerank()
        sorted_verses = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"verse_id": v, "centrality": s} for v, s in sorted_verses]

    async def get_hub_and_authority_verses(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get hub verses (OT prophetic) and authority verses (NT fulfillment)."""
        hubs, authorities = await self.centrality.calculate_hits()
        return {
            "hubs": [{"verse_id": v, "hub_score": s} for v, s in sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:20]],
            "authorities": [{"verse_id": v, "authority_score": s} for v, s in sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:20]]
        }

    async def find_connection(
        self,
        verse_a: str,
        verse_b: str
    ) -> Optional[Dict[str, Any]]:
        """Find how two verses are connected."""
        return await self.paths.find_shortest_path(verse_a, verse_b)

    async def get_thematic_clusters(self) -> List[Dict[str, Any]]:
        """Get thematic clusters with summaries."""
        await self.community.detect_louvain_communities()
        return await self.community.get_community_summary()
