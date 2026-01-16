"""
BIBLOS v2 - Hyper-Fractal Typology Engine Implementation Part 2

This file contains the implementation methods for the HyperFractalTypologyEngine.
It is imported by fractal_typology.py to complete the engine.

Contains:
- Layer analysis algorithms (WORD, PHRASE, VERSE, PERICOPE, CHAPTER, BOOK, COVENANTAL)
- Integration with Session 01 and Session 04
- Helper methods for pattern matching and scoring
- Neo4j storage methods
"""
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class HyperFractalTypologyEngineImpl:
    """Implementation methods for HyperFractalTypologyEngine."""

    async def analyze_word_layer(
        self, type_ref: str, antitype_ref: str
    ) -> List:
        """
        Find word-level typological correspondences.

        Detects:
        - Hebrew/Greek terms in TYPE_VOCABULARY database
        - Shared roots across testaments
        - LXX translation equivalences
        - Semantic field overlaps
        """
        from ml.engines.fractal_typology import (
            LayerConnection,
            TypologyLayer,
            TypeAntitypeRelation,
            CorrespondenceType,
            LAYER_CONFIG,
        )

        connections = []

        try:
            # Get lemmatized content words with semantic metadata
            type_data = await self.corpus.get_verse(type_ref)
            antitype_data = await self.corpus.get_verse(antitype_ref)

            if not type_data or not antitype_data:
                return connections

            type_words = type_data.get("words", [])
            antitype_words = antitype_data.get("words", [])

            # Extract lemmas
            type_lemmas = {w.get("lemma", w.get("text", "")) for w in type_words}
            antitype_lemmas = {w.get("lemma", w.get("text", "")) for w in antitype_words}

            # Check against known typological vocabulary
            for type_lemma in type_lemmas:
                if type_lemma in self.type_vocabulary:
                    pattern = self.type_vocabulary[type_lemma]

                    for antitype_lemma in antitype_lemmas:
                        # Check if this is a known type-antitype pairing
                        if antitype_lemma in pattern.get("antitype_terms", [type_lemma]):
                            strength = 0.8  # Known pattern baseline

                            conn = LayerConnection(
                                connection_id=f"{type_ref}:{antitype_ref}:WORD:{len(connections)}",
                                source_reference=type_ref,
                                target_reference=antitype_ref,
                                source_text=type_lemma,
                                target_text=antitype_lemma,
                                source_lemmas=[type_lemma],
                                target_lemmas=[antitype_lemma],
                                layer=TypologyLayer.WORD,
                                relation=TypeAntitypeRelation.PREFIGURATION,
                                correspondence_type=CorrespondenceType.LEXICAL,
                                correspondence_strength=strength,
                                semantic_similarity=0.9,
                                structural_similarity=0.0,
                                pattern_matches=[pattern["pattern_id"]],
                                explanation=f"'{type_lemma}' → '{antitype_lemma}' ({pattern['pattern_name']})",
                            )
                            connections.append(conn)

        except Exception as e:
            logger.warning(f"Error in word layer analysis: {e}")

        return connections

    async def analyze_phrase_layer(
        self, type_ref: str, antitype_ref: str
    ) -> List:
        """
        Find phrase-level typological correspondences.

        Detects:
        - Formulaic expressions
        - Title phrases ("Son of Man", "Lamb of God")
        - Action descriptions
        - Theological phrases
        """
        from ml.engines.fractal_typology import (
            LayerConnection,
            TypologyLayer,
            TypeAntitypeRelation,
            CorrespondenceType,
        )

        connections = []

        try:
            # For now, use verse-level text as phrases
            # In full implementation, would extract syntactic clauses
            type_data = await self.corpus.get_verse(type_ref)
            antitype_data = await self.corpus.get_verse(antitype_ref)

            if not type_data or not antitype_data:
                return connections

            type_text = type_data.get("text", "")
            antitype_text = antitype_data.get("text", "")

            # Simple semantic similarity check (placeholder)
            if type_text and antitype_text:
                # Would use embeddings in full implementation
                similarity = self._simple_text_similarity(type_text, antitype_text)

                if similarity > self.config.min_phrase_similarity:
                    conn = LayerConnection(
                        connection_id=f"{type_ref}:{antitype_ref}:PHRASE:0",
                        source_reference=type_ref,
                        target_reference=antitype_ref,
                        source_text=type_text[:100],
                        target_text=antitype_text[:100],
                        source_lemmas=[],
                        target_lemmas=[],
                        layer=TypologyLayer.PHRASE,
                        relation=TypeAntitypeRelation.PREFIGURATION,
                        correspondence_type=CorrespondenceType.SEMANTIC,
                        correspondence_strength=similarity,
                        semantic_similarity=similarity,
                        structural_similarity=0.5,
                        explanation=f"Phrase-level semantic parallel",
                    )
                    connections.append(conn)

        except Exception as e:
            logger.warning(f"Error in phrase layer analysis: {e}")

        return connections

    async def analyze_verse_layer(
        self, type_ref: str, antitype_ref: str
    ) -> List:
        """Find verse-level typological correspondences."""
        from ml.engines.fractal_typology import (
            LayerConnection,
            TypologyLayer,
            TypeAntitypeRelation,
            CorrespondenceType,
        )

        connections = []

        try:
            type_data = await self.corpus.get_verse(type_ref)
            antitype_data = await self.corpus.get_verse(antitype_ref)

            if not type_data or not antitype_data:
                return connections

            # Check if verses are in known type-antitype pairs
            for pattern in self.type_patterns.values():
                if type_ref in pattern.get_all_instances() and antitype_ref in pattern.get_all_instances():
                    conn = LayerConnection(
                        connection_id=f"{type_ref}:{antitype_ref}:VERSE:0",
                        source_reference=type_ref,
                        target_reference=antitype_ref,
                        source_text=type_data.get("text", "")[:100],
                        target_text=antitype_data.get("text", "")[:100],
                        source_lemmas=[],
                        target_lemmas=[],
                        layer=TypologyLayer.VERSE,
                        relation=pattern.relation_type,
                        correspondence_type=CorrespondenceType.FUNCTIONAL,
                        correspondence_strength=0.85,
                        semantic_similarity=0.8,
                        structural_similarity=0.7,
                        pattern_matches=[pattern.type_id],
                        explanation=f"Known typological pair: {pattern.pattern_name}",
                    )
                    connections.append(conn)
                    break

        except Exception as e:
            logger.warning(f"Error in verse layer analysis: {e}")

        return connections

    async def analyze_pericope_layer(
        self, type_pericope: str, antitype_pericope: str
    ) -> List:
        """
        Find narrative/pericope-level typological correspondences.

        Detects:
        - Narrative element parallels (actors, actions, outcomes)
        - Scene structure parallels
        - Discourse type matches
        """
        from ml.engines.fractal_typology import (
            LayerConnection,
            TypologyLayer,
            TypeAntitypeRelation,
            CorrespondenceType,
        )

        connections = []

        try:
            # Extract narrative elements (placeholder implementation)
            type_elements = await self.extract_narrative_elements(type_pericope)
            antitype_elements = await self.extract_narrative_elements(antitype_pericope)

            # Calculate element overlap
            total_similarity = 0.0
            for category in ["actors", "actions", "objects"]:
                type_set = set(type_elements.get(category, []))
                antitype_set = set(antitype_elements.get(category, []))

                if type_set and antitype_set:
                    overlap = len(type_set & antitype_set) / max(len(type_set | antitype_set), 1)
                    total_similarity += overlap

            total_similarity /= 3  # Average across categories

            if total_similarity > 0.4:
                conn = LayerConnection(
                    connection_id=f"{type_pericope}:{antitype_pericope}:PERICOPE:0",
                    source_reference=type_pericope,
                    target_reference=antitype_pericope,
                    source_text=f"Pericope: {type_pericope}",
                    target_text=f"Pericope: {antitype_pericope}",
                    source_lemmas=[],
                    target_lemmas=[],
                    layer=TypologyLayer.PERICOPE,
                    relation=TypeAntitypeRelation.PREFIGURATION,
                    correspondence_type=CorrespondenceType.STRUCTURAL,
                    correspondence_strength=total_similarity,
                    semantic_similarity=total_similarity,
                    structural_similarity=0.8,
                    explanation=f"Narrative parallel: {total_similarity:.0%} element overlap",
                )
                connections.append(conn)

        except Exception as e:
            logger.warning(f"Error in pericope layer analysis: {e}")

        return connections

    async def analyze_chapter_layer(
        self, type_ref: str, antitype_ref: str
    ) -> List:
        """Find chapter-level typological correspondences."""
        from ml.engines.fractal_typology import (
            LayerConnection,
            TypologyLayer,
            TypeAntitypeRelation,
            CorrespondenceType,
        )

        connections = []

        try:
            # Extract chapter references
            type_chapter = self._extract_chapter(type_ref)
            antitype_chapter = self._extract_chapter(antitype_ref)

            # Check for known chapter-level patterns
            # Placeholder: would check known correspondences like Genesis 22 → Hebrews 11

            if type_chapter and antitype_chapter:
                # Simple heuristic for now
                conn = LayerConnection(
                    connection_id=f"{type_ref}:{antitype_ref}:CHAPTER:0",
                    source_reference=type_chapter,
                    target_reference=antitype_chapter,
                    source_text=f"Chapter: {type_chapter}",
                    target_text=f"Chapter: {antitype_chapter}",
                    source_lemmas=[],
                    target_lemmas=[],
                    layer=TypologyLayer.CHAPTER,
                    relation=TypeAntitypeRelation.FULFILLMENT,
                    correspondence_type=CorrespondenceType.STRUCTURAL,
                    correspondence_strength=0.4,
                    semantic_similarity=0.5,
                    structural_similarity=0.6,
                    explanation=f"Chapter-level structural parallel",
                )
                connections.append(conn)

        except Exception as e:
            logger.warning(f"Error in chapter layer analysis: {e}")

        return connections

    async def analyze_book_layer(
        self, type_ref: str, antitype_ref: str
    ) -> List:
        """Find book-level typological correspondences."""
        from ml.engines.fractal_typology import (
            LayerConnection,
            TypologyLayer,
            TypeAntitypeRelation,
            CorrespondenceType,
        )

        connections = []

        try:
            type_book = self._extract_book(type_ref)
            antitype_book = self._extract_book(antitype_ref)

            # Check for known book-level typologies
            # Example: Genesis → John (beginnings, Logos)
            book_pairs = {
                ("GEN", "JHN"): ("Beginning themes, Logos",  0.7),
                ("EXO", "MAR"): ("Exodus/Gospel parallels", 0.6),
            }

            pair_key = (type_book, antitype_book)
            if pair_key in book_pairs:
                desc, strength = book_pairs[pair_key]
                conn = LayerConnection(
                    connection_id=f"{type_ref}:{antitype_ref}:BOOK:0",
                    source_reference=type_book,
                    target_reference=antitype_book,
                    source_text=f"Book: {type_book}",
                    target_text=f"Book: {antitype_book}",
                    source_lemmas=[],
                    target_lemmas=[],
                    layer=TypologyLayer.BOOK,
                    relation=TypeAntitypeRelation.FULFILLMENT,
                    correspondence_type=CorrespondenceType.FUNCTIONAL,
                    correspondence_strength=strength,
                    semantic_similarity=strength,
                    structural_similarity=0.5,
                    explanation=desc,
                )
                connections.append(conn)

        except Exception as e:
            logger.warning(f"Error in book layer analysis: {e}")

        return connections

    async def analyze_covenantal_layer(
        self, type_ref: str, antitype_ref: str
    ) -> List:
        """
        Find covenant-arc typological correspondences.

        Detects:
        - Promise-fulfillment across covenants
        - Covenant sign correspondences
        - Covenant mediator typology
        """
        from ml.engines.fractal_typology import (
            LayerConnection,
            TypologyLayer,
            TypeAntitypeRelation,
            CorrespondenceType,
        )

        connections = []

        try:
            # Determine covenant contexts
            type_covenant = await self.trace_covenant_arc(type_ref)
            antitype_covenant = await self.trace_covenant_arc(antitype_ref)

            if not type_covenant or not antitype_covenant:
                return connections

            # Check for promise overlap
            shared_promises = set(type_covenant.key_promises) & set(
                antitype_covenant.key_promises
            )

            for promise in shared_promises:
                conn = LayerConnection(
                    connection_id=f"{type_ref}:{antitype_ref}:COVENANTAL:{len(connections)}",
                    source_reference=type_ref,
                    target_reference=antitype_ref,
                    source_text=f"{type_covenant.covenant_name}: {promise}",
                    target_text=f"{antitype_covenant.covenant_name}: {promise}",
                    source_lemmas=[],
                    target_lemmas=[],
                    layer=TypologyLayer.COVENANTAL,
                    relation=TypeAntitypeRelation.FULFILLMENT,
                    correspondence_type=CorrespondenceType.FUNCTIONAL,
                    correspondence_strength=0.85,
                    semantic_similarity=0.9,
                    structural_similarity=0.8,
                    explanation=f"Covenant promise '{promise}' fulfillment",
                )
                connections.append(conn)

        except Exception as e:
            logger.warning(f"Error in covenantal layer analysis: {e}")

        return connections

    # Helper methods

    async def extract_narrative_elements(self, pericope_ref: str) -> Dict[str, List[str]]:
        """Extract narrative elements using discourse analysis (placeholder)."""
        elements = {
            "actors": [],
            "actions": [],
            "objects": [],
            "outcomes": [],
            "locations": [],
        }

        try:
            # Would use proper discourse analysis in full implementation
            pass
        except Exception as e:
            logger.warning(f"Error extracting narrative elements: {e}")

        return elements

    async def trace_covenant_arc(self, verse_ref: str):
        """Determine which covenant arc a verse belongs to."""
        # Check direct membership
        for covenant in self.covenant_arcs.values():
            if covenant.contains_reference(verse_ref):
                return covenant

        # Book-level association
        book = self._extract_book(verse_ref)
        book_covenant_map = {
            "GEN": "abrahamic",
            "EXO": "mosaic",
            "LEV": "mosaic",
            "2SA": "davidic",
            "PSA": "davidic",
            "ISA": "davidic",
            "JER": "new",
            "MAT": "new",
            "JHN": "new",
            "HEB": "new",
            "GAL": "abrahamic",
        }

        if book in book_covenant_map:
            return self.covenant_arcs.get(book_covenant_map[book])

        return None

    async def _expand_to_pericope(self, verse_ref: str) -> str:
        """Expand verse to pericope range (placeholder)."""
        # In full implementation, would use discourse markers
        # For now, just return the verse itself
        return verse_ref

    def _extract_book(self, ref: str) -> str:
        """Extract book code from reference."""
        parts = ref.split(".")
        return parts[0] if parts else ""

    def _extract_chapter(self, ref: str) -> str:
        """Extract book.chapter from reference."""
        parts = ref.split(".")
        return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else ref

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity (placeholder for embedding similarity)."""
        # Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def calculate_composite_strength(
        self, layers: Dict
    ) -> Tuple[float, "TypologyLayer"]:
        """Calculate weighted composite strength and identify dominant layer."""
        from ml.engines.fractal_typology import TypologyLayer, LAYER_CONFIG

        if not any(layers.values()):
            return 0.0, TypologyLayer.WORD

        layer_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for layer, connections in layers.items():
            if connections:
                # Use enriched strength
                strengths = [c.enriched_strength() for c in connections]
                avg_strength = sum(strengths) / len(strengths)
                max_strength = max(strengths)

                # Combine average and max
                layer_score = 0.7 * avg_strength + 0.3 * max_strength
                layer_scores[layer] = layer_score

                weight = LAYER_CONFIG[layer]["weight"]
                weighted_sum += layer_score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0, TypologyLayer.WORD

        base_score = weighted_sum / total_weight

        # Fractal depth bonus
        active_layers = len(layer_scores)
        depth_bonus = min(
            self.config.max_depth_bonus,
            active_layers * self.config.depth_bonus_per_layer,
        )

        # Patristic attestation bonus
        total_patristic = sum(
            len(c.patristic_attestation)
            for conns in layers.values()
            for c in conns
        )
        patristic_bonus = min(
            self.config.max_patristic_bonus,
            total_patristic * self.config.patristic_bonus_per_witness,
        )

        composite = min(1.0, base_score + depth_bonus + patristic_bonus)
        dominant = max(layer_scores.keys(), key=lambda k: layer_scores[k])

        return composite, dominant

    async def _enrich_with_transformation(self, layers: Dict) -> Dict:
        """Add mutual transformation scores from Session 01."""
        if not self.mutual_metric:
            return layers

        for layer, connections in layers.items():
            for conn in connections:
                try:
                    mt_result = await self.mutual_metric.measure_transformation(
                        source_verse=conn.source_reference,
                        target_verse=conn.target_reference,
                    )
                    conn.mutual_transformation = mt_result.mutual_influence
                except Exception as e:
                    logger.debug(f"MT enrichment failed: {e}")
                    conn.mutual_transformation = None
        return layers

    async def _enrich_with_necessity(self, layers: Dict) -> Dict:
        """Add necessity scores from Session 04."""
        if not self.necessity_calc:
            return layers

        for layer, connections in layers.items():
            for conn in connections:
                try:
                    necessity = await self.necessity_calc.calculate_necessity(
                        verse_a=conn.target_reference,  # NT needs OT
                        verse_b=conn.source_reference,
                    )
                    conn.necessity_score = necessity.necessity_score
                except Exception as e:
                    logger.debug(f"Necessity enrichment failed: {e}")
                    conn.necessity_score = None
        return layers

    def _identify_matched_patterns(self, layers: Dict) -> List:
        """Identify which TypePatterns this result instantiates."""
        matched = {}
        for layer, connections in layers.items():
            for conn in connections:
                for pattern_id in conn.pattern_matches:
                    if pattern_id in self.type_patterns:
                        matched[pattern_id] = self.type_patterns[pattern_id]

        return list(matched.values())

    def _build_reasoning_chain(self, layers: Dict, dominant) -> List[str]:
        """Build step-by-step typological reasoning."""
        chain = []

        chain.append(f"Dominant layer: {dominant.name}")

        for layer in layers.keys():
            conns = layers.get(layer, [])
            if conns:
                chain.append(f"{layer.name} layer: {len(conns)} connections found")

        return chain

    def _calculate_patristic_strength(self, layers: Dict) -> float:
        """Calculate aggregate patristic attestation score."""
        total_witnesses = 0
        for layer, connections in layers.items():
            for conn in connections:
                total_witnesses += len(conn.patristic_attestation)

        # Normalize to [0, 1]
        return min(1.0, total_witnesses * 0.1)

    def _aggregate_patristic_witnesses(self, layers: Dict) -> Dict[str, List[str]]:
        """Aggregate patristic witnesses across all connections."""
        witnesses = {}
        for layer, connections in layers.items():
            for conn in connections:
                for father, work, confidence in conn.patristic_attestation:
                    if father not in witnesses:
                        witnesses[father] = []
                    witnesses[father].append(work)

        return witnesses

    def _generate_synthesis(self, layers: Dict, matched_patterns: List) -> str:
        """Generate theological synthesis of findings."""
        synthesis = []

        if matched_patterns:
            pattern_names = [p.pattern_name for p in matched_patterns]
            synthesis.append(f"Patterns identified: {', '.join(pattern_names)}")

        active_layers = [layer.name for layer, conns in layers.items() if conns]
        synthesis.append(f"Active typological layers: {', '.join(active_layers)}")

        return " | ".join(synthesis) if synthesis else "Basic typological connection detected"

    def _calculate_confidence(self, layers: Dict, composite: float) -> float:
        """Calculate overall confidence score."""
        # Base confidence from composite strength
        confidence = composite

        # Boost for multiple layers
        active_count = sum(1 for conns in layers.values() if conns)
        if active_count >= 3:
            confidence = min(1.0, confidence + 0.1)

        return confidence

    def _calculate_confidence_interval(self, layers: Dict) -> Tuple[float, float]:
        """Calculate Bayesian credible interval (placeholder)."""
        # Would use proper Bayesian inference in full implementation
        # For now, return simple range based on layer count
        active_count = sum(1 for conns in layers.values() if conns)

        if active_count >= 3:
            return (0.6, 0.95)
        elif active_count >= 2:
            return (0.4, 0.8)
        else:
            return (0.2, 0.6)

    async def _store_in_neo4j(self, result) -> None:
        """Store typology result in Neo4j graph."""
        if not self.neo4j:
            return

        try:
            props = result.to_neo4j_properties()

            await self.neo4j.query(
                """
                MATCH (type:Verse {id: $type_ref})
                MATCH (antitype:Verse {id: $antitype_ref})
                MERGE (type)-[r:TYPIFIES]->(antitype)
                SET r += $props
                """,
                {
                    "type_ref": result.type_reference,
                    "antitype_ref": result.antitype_reference,
                    "props": props,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to store in Neo4j: {e}")
