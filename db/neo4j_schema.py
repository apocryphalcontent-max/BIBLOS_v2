"""
BIBLOS v2 - Neo4j SPIDERWEB Schema

Complete node and relationship schema for the graph-first biblical architecture.
Implements the interconnected web of biblical data: verses, words, meanings,
Church Fathers, covenants, prophecies, and typological patterns.
"""
from enum import Enum
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field


# =============================================================================
# NODE LABELS - Complete SPIDERWEB Taxonomy
# =============================================================================

class NodeLabel(Enum):
    """All node labels in the SPIDERWEB schema."""
    # Core textual nodes
    VERSE = "Verse"
    WORD = "Word"
    LEMMA = "Lemma"
    MEANING = "Meaning"

    # Patristic nodes
    FATHER = "Father"
    WORK = "Work"
    INTERPRETATION = "Interpretation"

    # Theological structure nodes
    PROPHECY = "Prophecy"
    COVENANT = "Covenant"
    TYPE_PATTERN = "TypePattern"
    SEMANTIC_DOMAIN = "SemanticDomain"
    LITURGICAL_CONTEXT = "LiturgicalContext"

    # Manuscript and textual tradition nodes
    MANUSCRIPT = "Manuscript"
    TEXTUAL_VARIANT = "TextualVariant"

    # Thematic and community nodes
    THEMATIC_CLUSTER = "ThematicCluster"
    BOOK = "Book"
    PERICOPE = "Pericope"

    @property
    def primary_key(self) -> str:
        """Primary key field for each node type."""
        return {
            NodeLabel.VERSE: "id",
            NodeLabel.WORD: "id",
            NodeLabel.LEMMA: "id",
            NodeLabel.MEANING: "id",
            NodeLabel.FATHER: "id",
            NodeLabel.WORK: "id",
            NodeLabel.INTERPRETATION: "id",
            NodeLabel.PROPHECY: "id",
            NodeLabel.COVENANT: "id",
            NodeLabel.TYPE_PATTERN: "id",
            NodeLabel.SEMANTIC_DOMAIN: "domain_id",
            NodeLabel.LITURGICAL_CONTEXT: "context_id",
            NodeLabel.MANUSCRIPT: "siglum",
            NodeLabel.TEXTUAL_VARIANT: "id",
            NodeLabel.THEMATIC_CLUSTER: "id",
            NodeLabel.BOOK: "code",
            NodeLabel.PERICOPE: "id",
        }[self]

    @property
    def required_indexes(self) -> List[str]:
        """Fields requiring secondary indexes for performance."""
        return {
            NodeLabel.VERSE: [
                "book", "testament", "chapter", "centrality_score",
                "community_id", "hub_score", "authority_score"
            ],
            NodeLabel.WORD: ["verse_id", "lemma_id", "position"],
            NodeLabel.LEMMA: ["language", "semantic_domain", "occurrence_count"],
            NodeLabel.MEANING: ["lemma_id", "theological_weight"],
            NodeLabel.FATHER: ["tradition", "era", "authority_weight"],
            NodeLabel.WORK: ["father_id", "approximate_date"],
            NodeLabel.PROPHECY: ["category", "bayesian_strength"],
            NodeLabel.COVENANT: ["progression_order", "status"],
            NodeLabel.TYPE_PATTERN: ["canonical_type", "canonical_antitype"],
            NodeLabel.MANUSCRIPT: ["type", "century", "priority_weight"],
            NodeLabel.BOOK: ["testament", "canonical_order"],
        }.get(self, [])

    @property
    def full_text_fields(self) -> List[str]:
        """Fields requiring full-text search indexes."""
        return {
            NodeLabel.VERSE: ["text_hebrew", "text_greek", "text_english"],
            NodeLabel.FATHER: ["name", "name_greek"],
            NodeLabel.WORK: ["title", "title_original"],
            NodeLabel.INTERPRETATION: ["content"],
        }.get(self, [])


# =============================================================================
# RELATIONSHIP TYPES - Complete Connection Taxonomy
# =============================================================================

class RelationshipType(Enum):
    """All relationship types in SPIDERWEB."""
    # Cross-reference relationships
    CROSS_REFERENCES = "CROSS_REFERENCES"
    VERBAL_PARALLEL = "VERBAL_PARALLEL"
    THEMATIC_PARALLEL = "THEMATIC_PARALLEL"
    STRUCTURAL_PARALLEL = "STRUCTURAL_PARALLEL"

    # Typological relationships
    TYPIFIES = "TYPIFIES"
    ANTITYPES = "ANTITYPES"
    SHADOWS = "SHADOWS"

    # Prophetic relationships
    FULFILLED_IN = "FULFILLED_IN"
    PROPHECY_OF = "PROPHECY_OF"
    PARTIALLY_FULFILLED_IN = "PARTIALLY_FULFILLED_IN"

    # Quotation relationships
    QUOTED_IN = "QUOTED_IN"
    ALLUDES_TO = "ALLUDES_TO"
    ECHOES = "ECHOES"

    # Linguistic relationships
    HAS_WORD = "HAS_WORD"
    HAS_LEMMA = "HAS_LEMMA"
    HAS_MEANING = "HAS_MEANING"
    SYNONYM_OF = "SYNONYM_OF"
    ANTONYM_OF = "ANTONYM_OF"

    # Patristic relationships
    INTERPRETS = "INTERPRETS"
    CITES = "CITES"
    AUTHORED = "AUTHORED"
    AGREES_WITH = "AGREES_WITH"
    DISPUTES = "DISPUTES"

    # Covenant relationships
    IN_COVENANT = "IN_COVENANT"
    BUILDS_ON = "BUILDS_ON"
    SUPERSEDES = "SUPERSEDES"
    RENEWS = "RENEWS"

    # Semantic relationships
    IN_SEMANTIC_DOMAIN = "IN_SEMANTIC_DOMAIN"
    RELATED_CONCEPT = "RELATED_CONCEPT"

    # Liturgical relationships
    USED_IN = "USED_IN"
    LITURGICAL_PAIR = "LITURGICAL_PAIR"

    # Manuscript relationships
    ATTESTED_IN = "ATTESTED_IN"
    VARIANT_OF = "VARIANT_OF"

    # Structural relationships
    IN_BOOK = "IN_BOOK"
    IN_PERICOPE = "IN_PERICOPE"
    FOLLOWS = "FOLLOWS"
    IN_CLUSTER = "IN_CLUSTER"

    @property
    def is_weighted(self) -> bool:
        """Whether this relationship carries a weight/confidence score."""
        weighted_types = {
            RelationshipType.CROSS_REFERENCES,
            RelationshipType.VERBAL_PARALLEL,
            RelationshipType.THEMATIC_PARALLEL,
            RelationshipType.STRUCTURAL_PARALLEL,
            RelationshipType.TYPIFIES,
            RelationshipType.ANTITYPES,
            RelationshipType.SHADOWS,
            RelationshipType.FULFILLED_IN,
            RelationshipType.PARTIALLY_FULFILLED_IN,
            RelationshipType.QUOTED_IN,
            RelationshipType.ALLUDES_TO,
            RelationshipType.ECHOES,
            RelationshipType.SYNONYM_OF,
            RelationshipType.AGREES_WITH,
            RelationshipType.RELATED_CONCEPT,
        }
        return self in weighted_types

    @property
    def weight_property(self) -> Optional[str]:
        """Name of the weight property if weighted."""
        if not self.is_weighted:
            return None
        weight_map = {
            RelationshipType.CROSS_REFERENCES: "confidence",
            RelationshipType.VERBAL_PARALLEL: "lexical_overlap",
            RelationshipType.THEMATIC_PARALLEL: "thematic_similarity",
            RelationshipType.STRUCTURAL_PARALLEL: "structural_score",
            RelationshipType.TYPIFIES: "composite_strength",
            RelationshipType.ANTITYPES: "composite_strength",
            RelationshipType.SHADOWS: "foreshadowing_score",
            RelationshipType.FULFILLED_IN: "necessity_score",
            RelationshipType.PARTIALLY_FULFILLED_IN: "fulfillment_degree",
            RelationshipType.QUOTED_IN: "match_confidence",
            RelationshipType.ALLUDES_TO: "allusion_strength",
            RelationshipType.ECHOES: "echo_strength",
            RelationshipType.SYNONYM_OF: "semantic_similarity",
            RelationshipType.AGREES_WITH: "agreement_degree",
            RelationshipType.RELATED_CONCEPT: "relatedness_score",
        }
        return weight_map.get(self, "weight")

    @property
    def is_directional(self) -> bool:
        """Whether direction matters for this relationship."""
        non_directional = {
            RelationshipType.VERBAL_PARALLEL,
            RelationshipType.THEMATIC_PARALLEL,
            RelationshipType.STRUCTURAL_PARALLEL,
            RelationshipType.SYNONYM_OF,
            RelationshipType.ANTONYM_OF,
            RelationshipType.AGREES_WITH,
            RelationshipType.RELATED_CONCEPT,
            RelationshipType.LITURGICAL_PAIR,
        }
        return self not in non_directional


# =============================================================================
# GRAPH METRICS - Algorithm Output Types
# =============================================================================

class GraphMetricType(Enum):
    """Metrics computed over the SPIDERWEB graph."""
    DEGREE_CENTRALITY = "degree_centrality"
    BETWEENNESS_CENTRALITY = "betweenness_centrality"
    CLOSENESS_CENTRALITY = "closeness_centrality"
    PAGERANK = "pagerank"
    ARTICLE_RANK = "article_rank"
    EIGENVECTOR_CENTRALITY = "eigenvector_centrality"
    CLUSTERING_COEFFICIENT = "clustering_coefficient"
    COMMUNITY_ID = "community_id"
    HUB_SCORE = "hub_score"
    AUTHORITY_SCORE = "authority_score"
    TRIANGLE_COUNT = "triangle_count"
    LOCAL_CLUSTERING = "local_clustering"

    @property
    def computation_cost(self) -> str:
        """GDS algorithm complexity class."""
        return {
            GraphMetricType.DEGREE_CENTRALITY: "O(V+E)",
            GraphMetricType.BETWEENNESS_CENTRALITY: "O(V*E)",
            GraphMetricType.CLOSENESS_CENTRALITY: "O(V*(V+E))",
            GraphMetricType.PAGERANK: "O(iterations*(V+E))",
            GraphMetricType.ARTICLE_RANK: "O(iterations*(V+E))",
            GraphMetricType.EIGENVECTOR_CENTRALITY: "O(iterations*(V+E))",
            GraphMetricType.CLUSTERING_COEFFICIENT: "O(V*d²)",
            GraphMetricType.COMMUNITY_ID: "O(V+E*log(V))",
            GraphMetricType.HUB_SCORE: "O(iterations*(V+E))",
            GraphMetricType.AUTHORITY_SCORE: "O(iterations*(V+E))",
            GraphMetricType.TRIANGLE_COUNT: "O(E^1.5)",
            GraphMetricType.LOCAL_CLUSTERING: "O(V*d²)",
        }[self]

    @property
    def requires_gds(self) -> bool:
        """Whether this metric requires GDS library."""
        non_gds = {
            GraphMetricType.DEGREE_CENTRALITY,
        }
        return self not in non_gds


class CommunityAlgorithm(Enum):
    """Community detection algorithms available in GDS."""
    LOUVAIN = "louvain"
    LABEL_PROPAGATION = "labelPropagation"
    WEAKLY_CONNECTED = "wcc"
    STRONGLY_CONNECTED = "scc"
    MODULARITY_OPTIMIZATION = "modularity"
    K_MEANS = "kmeans"
    LEIDEN = "leiden"

    @property
    def supports_weights(self) -> bool:
        """Whether algorithm supports relationship weights."""
        weighted = {
            CommunityAlgorithm.LOUVAIN,
            CommunityAlgorithm.LEIDEN,
            CommunityAlgorithm.MODULARITY_OPTIMIZATION,
        }
        return self in weighted


class PathAlgorithm(Enum):
    """Path finding algorithms."""
    SHORTEST_PATH = "shortestPath"
    ALL_SHORTEST_PATHS = "allShortestPaths"
    DIJKSTRA = "dijkstra"
    ASTAR = "astar"
    YELP = "yelp"
    RANDOM_WALK = "randomWalk"
    BFS = "bfs"
    DFS = "dfs"


# =============================================================================
# SCHEMA DATACLASSES - Node and Relationship Validation
# =============================================================================

@dataclass
class NodeSchema:
    """Schema definition for a node type."""
    label: NodeLabel
    properties: Dict[str, type]
    required_properties: List[str]
    computed_properties: List[str] = field(default_factory=list)
    full_text_fields: List[str] = field(default_factory=list)

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate data against schema."""
        for prop in self.required_properties:
            if prop not in data or data[prop] is None:
                return False
        return True

    def get_missing_required(self, data: Dict[str, Any]) -> List[str]:
        """Get list of missing required properties."""
        return [prop for prop in self.required_properties
                if prop not in data or data[prop] is None]


@dataclass
class RelationshipSchema:
    """Schema definition for a relationship type."""
    rel_type: RelationshipType
    source_labels: Set[NodeLabel]
    target_labels: Set[NodeLabel]
    properties: Dict[str, type]
    required_properties: List[str] = field(default_factory=list)

    def validate_endpoints(self, source_label: NodeLabel, target_label: NodeLabel) -> bool:
        """Validate source and target node labels."""
        return source_label in self.source_labels and target_label in self.target_labels


@dataclass
class GraphStatistics:
    """Global statistics for the SPIDERWEB graph."""
    node_count: int
    edge_count: int
    avg_degree: float
    density: float
    diameter: Optional[int]
    connected_components: int
    largest_component_size: int

    # Node type breakdown
    verse_count: int = 0
    word_count: int = 0
    lemma_count: int = 0
    father_count: int = 0

    # Relationship type breakdown
    cross_ref_count: int = 0
    typological_count: int = 0
    prophetic_count: int = 0
    patristic_count: int = 0

    @property
    def is_sparse(self) -> bool:
        """Biblical graphs are inherently sparse."""
        return self.density < 0.01

    @property
    def coverage_ratio(self) -> float:
        """Fraction of verses in largest component."""
        return self.largest_component_size / self.node_count if self.node_count else 0.0

    @property
    def is_well_connected(self) -> bool:
        """Check if graph is well-connected (>90% in main component)."""
        return self.coverage_ratio > 0.9


# =============================================================================
# PREDEFINED NODE SCHEMAS - Complete Schema Definitions
# =============================================================================

VERSE_SCHEMA = NodeSchema(
    label=NodeLabel.VERSE,
    properties={
        "id": str,              # GEN.1.1
        "book": str,            # GEN
        "book_full": str,       # Genesis
        "chapter": int,
        "verse": int,
        "testament": str,       # OT or NT
        "text_hebrew": str,
        "text_greek": str,
        "text_english": str,
        "word_count": int,
    },
    required_properties=["id", "book", "chapter", "verse", "testament"],
    computed_properties=[
        "centrality_score", "betweenness_score", "closeness_score",
        "community_id", "hub_score", "authority_score",
        "clustering_coefficient", "triangle_count", "processing_status"
    ],
    full_text_fields=["text_hebrew", "text_greek", "text_english"]
)

WORD_SCHEMA = NodeSchema(
    label=NodeLabel.WORD,
    properties={
        "id": str,              # GEN.1.1.1
        "verse_id": str,
        "position": int,
        "surface_form": str,
        "transliteration": str,
        "lemma_id": str,
        "part_of_speech": str,
        "morphology": str,
        "gloss": str,
        "syntactic_role": str,
        "clause_type": str,
        "discourse_function": str,
    },
    required_properties=["id", "verse_id", "position", "surface_form"]
)

LEMMA_SCHEMA = NodeSchema(
    label=NodeLabel.LEMMA,
    properties={
        "id": str,              # Strong's number (H7225, G3056)
        "language": str,
        "lemma": str,
        "transliteration": str,
        "gloss": str,
        "occurrence_count": int,
        "semantic_domain": str,
        "polysemous": bool,
        "meaning_count": int,
        "ot_occurrences": int,
        "nt_occurrences": int,
        "book_distribution": list,
    },
    required_properties=["id", "language", "lemma"]
)

MEANING_SCHEMA = NodeSchema(
    label=NodeLabel.MEANING,
    properties={
        "id": str,              # H7225.1
        "lemma_id": str,
        "meaning": str,
        "definition": str,
        "usage_count": int,
        "theological_weight": float,
        "example_verses": list,
        "syntactic_patterns": list,
        "collocations": list,
        "semantic_field": str,
    },
    required_properties=["id", "lemma_id", "meaning"]
)

FATHER_SCHEMA = NodeSchema(
    label=NodeLabel.FATHER,
    properties={
        "id": str,
        "name": str,
        "name_greek": str,
        "tradition": str,       # Eastern, Western, Syriac
        "era": str,
        "birth_year": int,
        "death_year": int,
        "notable_works": list,
        "authority_weight": float,
        "citation_count": int,
        "influence_score": float,
    },
    required_properties=["id", "name", "tradition"],
    computed_properties=["citation_count", "influence_score"]
)

WORK_SCHEMA = NodeSchema(
    label=NodeLabel.WORK,
    properties={
        "id": str,
        "father_id": str,
        "title": str,
        "title_original": str,
        "type": str,            # homily, commentary, treatise
        "book_count": int,
        "approximate_date": int,
        "verses_covered": int,
        "ot_coverage": float,
        "nt_coverage": float,
        "genre": str,
    },
    required_properties=["id", "father_id", "title"]
)

PROPHECY_SCHEMA = NodeSchema(
    label=NodeLabel.PROPHECY,
    properties={
        "id": str,
        "name": str,
        "prophecy_verse": str,
        "natural_probability": float,
        "independence_level": str,
        "category": str,        # messianic, national, eschatological
        "specificity_factors": list,
        "compound_probability": float,
        "bayesian_strength": str,
    },
    required_properties=["id", "name", "prophecy_verse"]
)

COVENANT_SCHEMA = NodeSchema(
    label=NodeLabel.COVENANT,
    properties={
        "id": str,
        "name": str,
        "initiation_verse": str,
        "key_promises": list,
        "conditional": bool,
        "status": str,          # active, superseded, eternal
        "promise_verses": list,
        "fulfillment_verses": list,
        "progression_order": int,
    },
    required_properties=["id", "name", "initiation_verse"]
)

TYPE_PATTERN_SCHEMA = NodeSchema(
    label=NodeLabel.TYPE_PATTERN,
    properties={
        "id": str,
        "name": str,
        "layers": list,         # WORD, PHRASE, VERSE, PERICOPE
        "canonical_type": str,
        "canonical_antitype": str,
        "keywords_hebrew": list,
        "keywords_greek": list,
        "correspondence_points": int,
        "fractal_depth": int,
    },
    required_properties=["id", "name", "canonical_type", "canonical_antitype"]
)

MANUSCRIPT_SCHEMA = NodeSchema(
    label=NodeLabel.MANUSCRIPT,
    properties={
        "siglum": str,          # B, א, A, etc.
        "name": str,
        "type": str,            # papyrus, uncial, minuscule
        "century": int,
        "contents": list,       # Books contained
        "priority_weight": float,
        "text_type": str,       # Alexandrian, Byzantine, Western
        "provenance": str,
    },
    required_properties=["siglum", "name", "type"]
)


# =============================================================================
# SCHEMA REGISTRY - Central Access Point
# =============================================================================

NODE_SCHEMAS: Dict[NodeLabel, NodeSchema] = {
    NodeLabel.VERSE: VERSE_SCHEMA,
    NodeLabel.WORD: WORD_SCHEMA,
    NodeLabel.LEMMA: LEMMA_SCHEMA,
    NodeLabel.MEANING: MEANING_SCHEMA,
    NodeLabel.FATHER: FATHER_SCHEMA,
    NodeLabel.WORK: WORK_SCHEMA,
    NodeLabel.PROPHECY: PROPHECY_SCHEMA,
    NodeLabel.COVENANT: COVENANT_SCHEMA,
    NodeLabel.TYPE_PATTERN: TYPE_PATTERN_SCHEMA,
    NodeLabel.MANUSCRIPT: MANUSCRIPT_SCHEMA,
}


def get_schema(label: NodeLabel) -> Optional[NodeSchema]:
    """Get schema for a node label."""
    return NODE_SCHEMAS.get(label)


def validate_node(label: NodeLabel, data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate node data against schema.

    Returns:
        Tuple of (is_valid, list_of_missing_fields)
    """
    schema = get_schema(label)
    if not schema:
        return True, []  # No schema defined, accept anything

    missing = schema.get_missing_required(data)
    return len(missing) == 0, missing


# =============================================================================
# CYPHER GENERATION UTILITIES
# =============================================================================

def generate_create_indexes_cypher() -> List[str]:
    """Generate Cypher statements to create all required indexes."""
    statements = []

    # Unique constraints (primary keys)
    for label in NodeLabel:
        pk = label.primary_key
        statements.append(
            f"CREATE CONSTRAINT {label.value.lower()}_{pk}_unique "
            f"IF NOT EXISTS FOR (n:{label.value}) REQUIRE n.{pk} IS UNIQUE"
        )

    # Performance indexes
    for label in NodeLabel:
        for field in label.required_indexes:
            idx_name = f"{label.value.lower()}_{field}_idx"
            statements.append(
                f"CREATE INDEX {idx_name} IF NOT EXISTS "
                f"FOR (n:{label.value}) ON (n.{field})"
            )

    # Full-text indexes
    for label in NodeLabel:
        ft_fields = label.full_text_fields
        if ft_fields:
            field_list = ", ".join([f"n.{f}" for f in ft_fields])
            statements.append(
                f"CREATE FULLTEXT INDEX {label.value.lower()}_fulltext "
                f"IF NOT EXISTS FOR (n:{label.value}) ON EACH [{field_list}]"
            )

    # Relationship property indexes
    for rel_type in RelationshipType:
        if rel_type.is_weighted:
            weight_prop = rel_type.weight_property
            if weight_prop:
                statements.append(
                    f"CREATE INDEX rel_{rel_type.value.lower()}_{weight_prop} "
                    f"IF NOT EXISTS FOR ()-[r:{rel_type.value}]-() ON (r.{weight_prop})"
                )

    return statements


def generate_merge_node_cypher(label: NodeLabel, data: Dict[str, Any]) -> tuple[str, Dict]:
    """Generate MERGE statement for creating/updating a node.

    Returns:
        Tuple of (cypher_statement, parameters)
    """
    pk = label.primary_key
    pk_value = data.get(pk)

    if pk_value is None:
        raise ValueError(f"Primary key '{pk}' is required for {label.value}")

    cypher = f"""
        MERGE (n:{label.value} {{{pk}: $pk_value}})
        SET n += $props
        RETURN n
    """

    params = {
        "pk_value": pk_value,
        "props": {k: v for k, v in data.items() if v is not None}
    }

    return cypher, params


def generate_create_relationship_cypher(
    rel_type: RelationshipType,
    source_label: NodeLabel,
    source_key: str,
    target_label: NodeLabel,
    target_key: str,
    properties: Optional[Dict[str, Any]] = None
) -> tuple[str, Dict]:
    """Generate CREATE statement for a relationship.

    Returns:
        Tuple of (cypher_statement, parameters)
    """
    props = properties or {}

    cypher = f"""
        MATCH (source:{source_label.value} {{{source_label.primary_key}: $source_key}})
        MATCH (target:{target_label.value} {{{target_label.primary_key}: $target_key}})
        CREATE (source)-[r:{rel_type.value}]->(target)
        SET r += $props
        RETURN r
    """

    params = {
        "source_key": source_key,
        "target_key": target_key,
        "props": props
    }

    return cypher, params
