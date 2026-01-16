"""
BIBLOS v2 - SYNTAKTIKOS Agent

Syntactic parsing and dependency analysis for biblical texts.
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import re

from agents.base import (
    BaseExtractionAgent,
    AgentConfig,
    ExtractionResult,
    ExtractionType,
    ProcessingStatus
)


class DependencyRelation(Enum):
    """Dependency relation types."""
    ROOT = "root"
    SUBJECT = "nsubj"
    OBJECT = "obj"
    INDIRECT_OBJECT = "iobj"
    PREDICATE = "pred"
    MODIFIER = "mod"
    DETERMINER = "det"
    PREPOSITION = "prep"
    COMPLEMENT = "comp"
    CONJUNCTION = "conj"
    ADVERBIAL = "advmod"
    GENITIVE = "gen"
    APPOSITION = "appos"
    VOCATIVE = "voc"


@dataclass
class DependencyNode:
    """A node in the dependency tree."""
    index: int
    word: str
    head: int  # Index of head word (-1 for root)
    relation: DependencyRelation
    children: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "word": self.word,
            "head": self.head,
            "relation": self.relation.value,
            "children": self.children
        }


@dataclass
class ClauseStructure:
    """Structure of a clause."""
    clause_type: str  # main, subordinate, relative, conditional
    verb: Optional[str]
    subject: Optional[str]
    objects: List[str]
    modifiers: List[str]


class SyntaktikosAgent(BaseExtractionAgent):
    """
    SYNTAKTIKOS - Syntactic analysis agent.

    Performs:
    - Dependency parsing
    - Clause structure analysis
    - Word order analysis
    - Syntactic pattern identification
    """

    # Greek word order patterns
    GREEK_PATTERNS = {
        "VSO": "verb-subject-object",
        "SVO": "subject-verb-object",
        "OVS": "object-verb-subject (emphatic)",
        "SOV": "subject-object-verb"
    }

    # Subordinating conjunctions
    SUBORDINATORS = {
        "greek": ["ὅτι", "ἵνα", "ὅταν", "ἐάν", "εἰ", "ὡς", "διότι", "καθώς"],
        "hebrew": ["כי", "אשר", "אם", "כאשר", "למען"],
        "english": ["that", "because", "when", "if", "so that", "as", "while"]
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="syntaktikos",
                extraction_type=ExtractionType.SYNTACTIC,
                batch_size=500,
                min_confidence=0.65
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.syntaktikos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Extract syntactic analysis from verse."""
        # Get morphological data if available
        morph_data = context.get("linguistic_results", {}).get("morphologos", {})

        # Parse dependencies
        tokens = self._tokenize(text)
        dependencies = self._parse_dependencies(tokens, morph_data)

        # Analyze clause structure
        clauses = self._analyze_clauses(text, dependencies)

        # Identify word order pattern
        word_order = self._identify_word_order(dependencies)

        # Identify syntactic patterns
        patterns = self._identify_patterns(text, clauses)

        data = {
            "tokens": tokens,
            "dependencies": [d.to_dict() for d in dependencies],
            "clauses": [self._clause_to_dict(c) for c in clauses],
            "word_order": word_order,
            "patterns": patterns,
            "tree_depth": self._calculate_tree_depth(dependencies),
            "complexity_score": self._calculate_complexity(clauses, dependencies)
        }

        confidence = self._calculate_confidence(dependencies, clauses)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        pattern = r'[\w\u0370-\u03FF\u1F00-\u1FFF\u0590-\u05FF]+'
        return re.findall(pattern, text)

    def _parse_dependencies(
        self,
        tokens: List[str],
        morph_data: Dict[str, Any]
    ) -> List[DependencyNode]:
        """Parse dependency structure."""
        if not tokens:
            return []

        nodes = []
        root_idx = self._find_root(tokens, morph_data)

        for i, token in enumerate(tokens):
            if i == root_idx:
                relation = DependencyRelation.ROOT
                head = -1
            else:
                relation, head = self._determine_relation(
                    i, tokens, root_idx, morph_data
                )

            nodes.append(DependencyNode(
                index=i,
                word=token,
                head=head,
                relation=relation,
                children=[]
            ))

        # Build children lists
        for node in nodes:
            if node.head >= 0:
                nodes[node.head].children.append(node.index)

        return nodes

    def _find_root(
        self,
        tokens: List[str],
        morph_data: Dict[str, Any]
    ) -> int:
        """Find the root (main verb) of the sentence."""
        # Check morphological data for verbs
        analyses = morph_data.get("data", {}).get("analyses", [])

        for i, analysis in enumerate(analyses):
            if analysis and analysis.get("analysis", {}).get("pos") == "verb":
                mood = analysis.get("analysis", {}).get("mood")
                if mood in ["indicative", "imperative"]:
                    return i

        # Default: first content word
        return 0 if tokens else -1

    def _determine_relation(
        self,
        token_idx: int,
        tokens: List[str],
        root_idx: int,
        morph_data: Dict[str, Any]
    ) -> Tuple[DependencyRelation, int]:
        """Determine dependency relation and head for a token."""
        # Simplified heuristics
        analyses = morph_data.get("data", {}).get("analyses", [])

        if token_idx < len(analyses) and analyses[token_idx]:
            analysis = analyses[token_idx].get("analysis", {})
            pos = analysis.get("pos")
            case = analysis.get("case")

            if pos == "article":
                # Article attaches to following noun
                return DependencyRelation.DETERMINER, min(token_idx + 1, len(tokens) - 1)

            if pos == "preposition":
                return DependencyRelation.PREPOSITION, root_idx

            if case == "nominative":
                return DependencyRelation.SUBJECT, root_idx

            if case == "accusative":
                return DependencyRelation.OBJECT, root_idx

            if case == "genitive":
                # Attach to preceding noun
                return DependencyRelation.GENITIVE, max(0, token_idx - 1)

            if case == "dative":
                return DependencyRelation.INDIRECT_OBJECT, root_idx

        # Default: attach to root as modifier
        return DependencyRelation.MODIFIER, root_idx

    def _analyze_clauses(
        self,
        text: str,
        dependencies: List[DependencyNode]
    ) -> List[ClauseStructure]:
        """Analyze clause structure."""
        clauses = []

        # Split on clause boundaries
        clause_texts = re.split(r'[,;:]|\bκαί\b|\bוְ\b', text)

        for clause_text in clause_texts:
            if not clause_text.strip():
                continue

            clause_type = self._determine_clause_type(clause_text)

            # Extract constituents from dependencies
            verb = None
            subject = None
            objects = []
            modifiers = []

            clause_tokens = set(self._tokenize(clause_text))

            for node in dependencies:
                if node.word not in clause_tokens:
                    continue

                if node.relation == DependencyRelation.ROOT:
                    verb = node.word
                elif node.relation == DependencyRelation.SUBJECT:
                    subject = node.word
                elif node.relation in [DependencyRelation.OBJECT, DependencyRelation.INDIRECT_OBJECT]:
                    objects.append(node.word)
                elif node.relation in [DependencyRelation.MODIFIER, DependencyRelation.ADVERBIAL]:
                    modifiers.append(node.word)

            clauses.append(ClauseStructure(
                clause_type=clause_type,
                verb=verb,
                subject=subject,
                objects=objects,
                modifiers=modifiers
            ))

        return clauses

    def _determine_clause_type(self, text: str) -> str:
        """Determine type of clause."""
        text_lower = text.lower()

        # Check for subordinating conjunctions
        for lang, conjunctions in self.SUBORDINATORS.items():
            for conj in conjunctions:
                if conj in text_lower:
                    if conj in ["ἵνα", "למען", "so that"]:
                        return "purpose"
                    if conj in ["ὅτι", "כי", "because", "διότι"]:
                        return "causal"
                    if conj in ["ἐάν", "εἰ", "אם", "if"]:
                        return "conditional"
                    if conj in ["ὅταν", "כאשר", "when"]:
                        return "temporal"
                    return "subordinate"

        # Check for relative clauses
        if any(rel in text_lower for rel in ["ὅς", "ἥ", "ὅ", "אשר", "who", "which", "that"]):
            return "relative"

        return "main"

    def _identify_word_order(
        self,
        dependencies: List[DependencyNode]
    ) -> str:
        """Identify word order pattern."""
        positions = {"V": -1, "S": -1, "O": -1}

        for node in dependencies:
            if node.relation == DependencyRelation.ROOT:
                positions["V"] = node.index
            elif node.relation == DependencyRelation.SUBJECT:
                positions["S"] = node.index
            elif node.relation == DependencyRelation.OBJECT:
                positions["O"] = node.index

        if all(pos >= 0 for pos in positions.values()):
            order = sorted(positions.items(), key=lambda x: x[1])
            return "".join(item[0] for item in order)

        return "undetermined"

    def _identify_patterns(
        self,
        text: str,
        clauses: List[ClauseStructure]
    ) -> List[str]:
        """Identify syntactic patterns."""
        patterns = []

        # Check for parallelism
        if len(clauses) >= 2:
            structures = [
                (c.clause_type, bool(c.verb), bool(c.subject))
                for c in clauses
            ]
            if len(set(structures)) == 1:
                patterns.append("parallel_structure")

        # Check for chiasm
        if len(clauses) >= 4:
            # Simplified chiasm detection
            patterns.append("potential_chiasm")

        # Check for inclusio
        words = self._tokenize(text)
        if len(words) >= 6:
            if words[0].lower() == words[-1].lower():
                patterns.append("inclusio")

        return patterns

    def _clause_to_dict(self, clause: ClauseStructure) -> Dict[str, Any]:
        """Convert clause to dictionary."""
        return {
            "clause_type": clause.clause_type,
            "verb": clause.verb,
            "subject": clause.subject,
            "objects": clause.objects,
            "modifiers": clause.modifiers
        }

    def _calculate_tree_depth(self, dependencies: List[DependencyNode]) -> int:
        """Calculate maximum depth of dependency tree."""
        if not dependencies:
            return 0

        def get_depth(idx: int, visited: set) -> int:
            if idx < 0 or idx >= len(dependencies) or idx in visited:
                return 0
            visited.add(idx)
            node = dependencies[idx]
            if not node.children:
                return 1
            return 1 + max(get_depth(c, visited.copy()) for c in node.children)

        # Find root
        root_idx = next(
            (i for i, n in enumerate(dependencies) if n.head == -1),
            0
        )
        return get_depth(root_idx, set())

    def _calculate_complexity(
        self,
        clauses: List[ClauseStructure],
        dependencies: List[DependencyNode]
    ) -> float:
        """Calculate syntactic complexity score."""
        if not dependencies:
            return 0.0

        # Factors: number of clauses, tree depth, subordination
        clause_factor = len(clauses) / 5.0  # Normalize
        depth_factor = self._calculate_tree_depth(dependencies) / 10.0
        subordinate_count = sum(
            1 for c in clauses if c.clause_type != "main"
        )
        subordination_factor = subordinate_count / max(1, len(clauses))

        return min(1.0, (clause_factor + depth_factor + subordination_factor) / 3)

    def _calculate_confidence(
        self,
        dependencies: List[DependencyNode],
        clauses: List[ClauseStructure]
    ) -> float:
        """Calculate confidence score."""
        if not dependencies:
            return 0.0

        # Check for root
        has_root = any(d.relation == DependencyRelation.ROOT for d in dependencies)
        root_score = 0.3 if has_root else 0.0

        # Check for subject
        has_subject = any(d.relation == DependencyRelation.SUBJECT for d in dependencies)
        subject_score = 0.2 if has_subject else 0.0

        # Clause analysis quality
        clause_score = 0.3 if clauses else 0.0

        # Base confidence
        base = 0.2

        return min(1.0, base + root_score + subject_score + clause_score)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "dependencies" in data and "clauses" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus", "morphologos"]
