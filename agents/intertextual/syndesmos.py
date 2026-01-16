"""
BIBLOS v2 - SYNDESMOS Agent

Cross-reference connection analysis for biblical texts.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from agents.base import (
    BaseExtractionAgent,
    AgentConfig,
    ExtractionResult,
    ExtractionType,
    ProcessingStatus,
)
from data.schemas import ConnectionType, ConnectionStrength


@dataclass
class CrossReference:
    """A cross-reference connection."""
    source_ref: str
    target_ref: str
    connection_type: ConnectionType
    strength: ConnectionStrength
    shared_terms: List[str]
    explanation: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "target_ref": self.target_ref,
            "connection_type": self.connection_type.value,
            "strength": self.strength.value,
            "shared_terms": self.shared_terms,
            "explanation": self.explanation,
            "confidence": self.confidence
        }


class SyndesmosAgent(BaseExtractionAgent):
    """
    SYNDESMOS - Cross-reference analysis agent.

    Performs:
    - Cross-reference identification
    - Connection type classification
    - Strength assessment
    - Network analysis
    - Cluster identification
    """

    # Thematic keywords for connection detection
    THEMATIC_CLUSTERS = {
        "creation": ["create", "made", "beginning", "heaven", "earth", "light"],
        "covenant": ["covenant", "promise", "oath", "sworn", "faithful"],
        "redemption": ["redeem", "deliver", "save", "ransom", "rescue"],
        "kingdom": ["king", "kingdom", "reign", "throne", "rule"],
        "temple": ["temple", "house", "sanctuary", "dwelling", "tabernacle"],
        "sacrifice": ["sacrifice", "offering", "blood", "altar", "lamb"],
        "law": ["law", "commandment", "statute", "ordinance", "decree"],
        "faith": ["faith", "believe", "trust", "faithful", "obey"],
        "love": ["love", "mercy", "compassion", "kindness", "lovingkindness"],
        "judgment": ["judge", "judgment", "justice", "righteous", "wrath"]
    }

    # Key verbal connections
    VERBAL_MARKERS = {
        "quotation": ["it is written", "scripture says", "as it says"],
        "fulfillment": ["fulfilled", "according to", "that it might be fulfilled"],
        "allusion": ["like", "as", "in the same way"],
        "echo": ["remember", "recall", "consider"]
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="syndesmos",
                extraction_type=ExtractionType.INTERTEXTUAL,
                batch_size=200,
                min_confidence=0.6
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.syndesmos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Extract cross-reference analysis from verse."""
        # Get existing cross-references from context
        known_refs = context.get("cross_references", [])

        # Analyze existing connections
        analyzed_refs = self._analyze_connections(verse_id, known_refs, text)

        # Identify potential new connections
        potential = self._identify_potential(text, verse_id, context)

        # Classify connection types
        typed_refs = self._classify_types(analyzed_refs, text)

        # Build network metrics
        network = self._build_network_metrics(typed_refs)

        # Identify clusters
        clusters = self._identify_clusters(typed_refs)

        data = {
            "cross_references": [r.to_dict() for r in typed_refs],
            "potential_connections": potential,
            "network_metrics": network,
            "clusters": clusters,
            "connection_summary": self._summarize_connections(typed_refs),
            "intertextual_density": self._calculate_density(typed_refs)
        }

        confidence = self._calculate_confidence(typed_refs, known_refs)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _analyze_connections(
        self,
        verse_id: str,
        known_refs: List[Dict[str, Any]],
        text: str
    ) -> List[CrossReference]:
        """Analyze known cross-references."""
        connections = []

        for ref in known_refs:
            target = ref.get("target_ref", ref.get("reference", ""))
            conn_type = self._determine_connection_type(
                ref.get("connection_type", "thematic")
            )
            strength = self._assess_strength(ref, text)
            shared = ref.get("shared_terms", [])

            if isinstance(shared, str):
                shared = [shared]

            connections.append(CrossReference(
                source_ref=verse_id,
                target_ref=target,
                connection_type=conn_type,
                strength=strength,
                shared_terms=shared,
                explanation=ref.get("explanation", ""),
                confidence=ref.get("confidence", 0.7)
            ))

        return connections

    def _determine_connection_type(self, type_str: str) -> ConnectionType:
        """Determine connection type from string."""
        type_map = {
            "thematic": ConnectionType.THEMATIC,
            "verbal": ConnectionType.VERBAL,
            "conceptual": ConnectionType.CONCEPTUAL,
            "historical": ConnectionType.HISTORICAL,
            "typological": ConnectionType.TYPOLOGICAL,
            "prophetic": ConnectionType.PROPHETIC,
            "liturgical": ConnectionType.LITURGICAL,
            "narrative": ConnectionType.NARRATIVE,
            "genealogical": ConnectionType.GENEALOGICAL,
            "geographical": ConnectionType.GEOGRAPHICAL
        }
        return type_map.get(type_str.lower(), ConnectionType.THEMATIC)

    def _assess_strength(
        self,
        ref: Dict[str, Any],
        text: str
    ) -> ConnectionStrength:
        """Assess connection strength."""
        # Check for explicit markers
        text_lower = text.lower()

        for marker in self.VERBAL_MARKERS["quotation"]:
            if marker in text_lower:
                return ConnectionStrength.STRONG

        # Check existing strength
        existing = ref.get("strength", "moderate").lower()
        if existing == "strong":
            return ConnectionStrength.STRONG
        elif existing == "weak":
            return ConnectionStrength.WEAK

        return ConnectionStrength.MODERATE

    def _identify_potential(
        self,
        text: str,
        verse_id: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify potential new connections."""
        potential = []
        text_lower = text.lower()

        # Check thematic clusters
        for theme, keywords in self.THEMATIC_CLUSTERS.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if len(matches) >= 2:
                potential.append({
                    "theme": theme,
                    "keywords": matches,
                    "suggestion": f"Check other {theme} passages"
                })

        # Check for quotation/allusion markers
        for marker_type, markers in self.VERBAL_MARKERS.items():
            for marker in markers:
                if marker in text_lower:
                    potential.append({
                        "type": marker_type,
                        "marker": marker,
                        "suggestion": f"Investigate {marker_type} source"
                    })

        return potential

    def _classify_types(
        self,
        refs: List[CrossReference],
        text: str
    ) -> List[CrossReference]:
        """Further classify connection types based on text analysis."""
        text_lower = text.lower()

        for ref in refs:
            # Enhance classification
            if ref.connection_type == ConnectionType.THEMATIC:
                # Check for more specific type
                if any(kw in text_lower for kw in ["fulfilled", "prophecy"]):
                    ref.connection_type = ConnectionType.PROPHETIC
                elif any(kw in text_lower for kw in ["type", "shadow", "figure"]):
                    ref.connection_type = ConnectionType.TYPOLOGICAL

        return refs

    def _build_network_metrics(
        self,
        refs: List[CrossReference]
    ) -> Dict[str, Any]:
        """Build network metrics for connections."""
        if not refs:
            return {
                "node_count": 0,
                "edge_count": 0,
                "density": 0.0
            }

        # Count unique nodes
        nodes = set()
        for ref in refs:
            nodes.add(ref.source_ref)
            nodes.add(ref.target_ref)

        return {
            "node_count": len(nodes),
            "edge_count": len(refs),
            "density": len(refs) / max(1, len(nodes) * (len(nodes) - 1) / 2),
            "avg_strength": sum(
                1.0 if r.strength == ConnectionStrength.STRONG else
                0.5 if r.strength == ConnectionStrength.MODERATE else 0.3
                for r in refs
            ) / max(1, len(refs))
        }

    def _identify_clusters(
        self,
        refs: List[CrossReference]
    ) -> List[Dict[str, Any]]:
        """Identify thematic clusters in connections."""
        clusters = {}

        for ref in refs:
            conn_type = ref.connection_type.value
            if conn_type not in clusters:
                clusters[conn_type] = {
                    "type": conn_type,
                    "count": 0,
                    "references": []
                }
            clusters[conn_type]["count"] += 1
            clusters[conn_type]["references"].append(ref.target_ref)

        return list(clusters.values())

    def _summarize_connections(
        self,
        refs: List[CrossReference]
    ) -> Dict[str, Any]:
        """Summarize connection analysis."""
        if not refs:
            return {
                "total": 0,
                "by_type": {},
                "by_strength": {}
            }

        by_type = {}
        by_strength = {}

        for ref in refs:
            t = ref.connection_type.value
            s = ref.strength.value
            by_type[t] = by_type.get(t, 0) + 1
            by_strength[s] = by_strength.get(s, 0) + 1

        return {
            "total": len(refs),
            "by_type": by_type,
            "by_strength": by_strength,
            "primary_type": max(by_type, key=by_type.get) if by_type else None
        }

    def _calculate_density(
        self,
        refs: List[CrossReference]
    ) -> float:
        """Calculate intertextual density."""
        if not refs:
            return 0.0

        # Density based on count and strength
        base = len(refs) * 0.1
        strength_bonus = sum(
            0.1 if r.strength == ConnectionStrength.STRONG else 0.05
            for r in refs
        )

        return min(1.0, base + strength_bonus + 0.2)

    def _calculate_confidence(
        self,
        typed_refs: List[CrossReference],
        known_refs: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score."""
        confidence = 0.5

        if known_refs:
            confidence += 0.2

        if typed_refs:
            avg_conf = sum(r.confidence for r in typed_refs) / len(typed_refs)
            confidence += avg_conf * 0.3

        return min(1.0, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "cross_references" in data and "network_metrics" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus", "semantikos", "typologos"]
