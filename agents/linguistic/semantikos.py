"""
BIBLOS v2 - SEMANTIKOS Agent

Semantic analysis and role labeling for biblical texts.
"""
from typing import Dict, List, Any, Optional, Set
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


class SemanticRole(Enum):
    """Semantic roles (thematic roles)."""
    AGENT = "agent"  # Doer of action
    PATIENT = "patient"  # Entity affected by action
    THEME = "theme"  # Entity moved/changed
    EXPERIENCER = "experiencer"  # Entity experiencing
    BENEFICIARY = "beneficiary"  # Entity benefiting
    INSTRUMENT = "instrument"  # Means of action
    LOCATION = "location"  # Where action occurs
    SOURCE = "source"  # Origin of movement
    GOAL = "goal"  # Destination of movement
    TIME = "time"  # When action occurs
    MANNER = "manner"  # How action is performed
    CAUSE = "cause"  # Why action occurs
    PURPOSE = "purpose"  # Intended result
    RECIPIENT = "recipient"  # Receiver
    STIMULUS = "stimulus"  # What causes experience


class SemanticDomain(Enum):
    """Semantic domains for biblical vocabulary."""
    DIVINE = "divine"
    HUMAN = "human"
    CREATION = "creation"
    COVENANT = "covenant"
    LAW = "law"
    WORSHIP = "worship"
    SIN = "sin"
    SALVATION = "salvation"
    ESCHATOLOGY = "eschatology"
    ETHICS = "ethics"
    FAMILY = "family"
    WARFARE = "warfare"
    KINGSHIP = "kingship"
    PROPHECY = "prophecy"
    WISDOM = "wisdom"


@dataclass
class SemanticFrame:
    """A semantic frame representing an event/situation."""
    predicate: str
    frame_type: str  # action, state, process, achievement
    roles: Dict[SemanticRole, str]
    domain: Optional[SemanticDomain]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "frame_type": self.frame_type,
            "roles": {role.value: filler for role, filler in self.roles.items()},
            "domain": self.domain.value if self.domain else None
        }


class SemantikosAgent(BaseExtractionAgent):
    """
    SEMANTIKOS - Semantic analysis agent.

    Performs:
    - Semantic role labeling
    - Semantic domain classification
    - Frame semantic analysis
    - Word sense disambiguation
    - Semantic relationship extraction
    """

    # Domain keywords
    DOMAIN_KEYWORDS = {
        SemanticDomain.DIVINE: [
            "θεός", "κύριος", "אלהים", "יהוה", "god", "lord", "spirit", "holy",
            "glory", "almighty", "eternal", "heaven"
        ],
        SemanticDomain.COVENANT: [
            "διαθήκη", "ברית", "covenant", "promise", "oath", "sworn",
            "faithful", "steadfast", "lovingkindness"
        ],
        SemanticDomain.SALVATION: [
            "σωτηρία", "λύτρωσις", "ישועה", "save", "redeem", "deliver",
            "rescue", "salvation", "ransom", "forgive"
        ],
        SemanticDomain.SIN: [
            "ἁμαρτία", "חטא", "sin", "transgress", "iniquity", "guilt",
            "wicked", "evil", "unrighteousness"
        ],
        SemanticDomain.WORSHIP: [
            "προσκυνέω", "השתחוה", "worship", "praise", "sacrifice",
            "offering", "temple", "altar", "priest"
        ],
        SemanticDomain.PROPHECY: [
            "προφητεία", "נבואה", "prophet", "vision", "oracle",
            "foretell", "reveal", "word of the lord"
        ],
        SemanticDomain.KINGSHIP: [
            "βασιλεύς", "מלך", "king", "throne", "reign", "kingdom",
            "crown", "scepter", "sovereign"
        ]
    }

    # Verbs that indicate specific semantic frames
    FRAME_VERBS = {
        "speech": ["λέγω", "εἶπον", "אמר", "say", "speak", "tell", "declare"],
        "motion": ["ἔρχομαι", "πορεύομαι", "הלך", "בוא", "go", "come", "walk"],
        "perception": ["βλέπω", "ὁράω", "ראה", "see", "hear", "perceive"],
        "cognition": ["γινώσκω", "οἶδα", "ידע", "know", "understand", "think"],
        "giving": ["δίδωμι", "נתן", "give", "grant", "bestow"],
        "causation": ["ποιέω", "עשה", "make", "cause", "create"],
        "possession": ["ἔχω", "have", "possess", "own", "hold"]
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="semantikos",
                extraction_type=ExtractionType.SEMANTIC,
                batch_size=500,
                min_confidence=0.65
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.semantikos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Extract semantic analysis from verse."""
        # Get syntactic data if available
        syntax_data = context.get("linguistic_results", {}).get("syntaktikos", {})

        # Extract semantic frames
        frames = self._extract_frames(text, syntax_data)

        # Identify semantic domains
        domains = self._identify_domains(text)

        # Label semantic roles
        roles = self._label_roles(text, syntax_data)

        # Extract semantic relationships
        relationships = self._extract_relationships(text, frames)

        data = {
            "frames": [f.to_dict() for f in frames],
            "domains": [d.value for d in domains],
            "role_assignments": roles,
            "relationships": relationships,
            "key_concepts": self._extract_key_concepts(text, domains),
            "semantic_density": self._calculate_density(frames, domains)
        }

        confidence = self._calculate_confidence(frames, domains, roles)

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
        return re.findall(pattern, text.lower())

    def _extract_frames(
        self,
        text: str,
        syntax_data: Dict[str, Any]
    ) -> List[SemanticFrame]:
        """Extract semantic frames from text."""
        frames = []
        tokens = self._tokenize(text)

        # Find frame-evoking predicates
        for frame_type, verbs in self.FRAME_VERBS.items():
            for verb in verbs:
                if verb.lower() in [t.lower() for t in tokens]:
                    # Build frame
                    roles = self._assign_roles_to_frame(
                        verb, frame_type, text, syntax_data
                    )
                    domain = self._determine_frame_domain(text, frame_type)

                    frames.append(SemanticFrame(
                        predicate=verb,
                        frame_type=frame_type,
                        roles=roles,
                        domain=domain
                    ))

        return frames

    def _assign_roles_to_frame(
        self,
        predicate: str,
        frame_type: str,
        text: str,
        syntax_data: Dict[str, Any]
    ) -> Dict[SemanticRole, str]:
        """Assign semantic roles based on frame type."""
        roles: Dict[SemanticRole, str] = {}

        # Get dependencies if available
        deps = syntax_data.get("data", {}).get("dependencies", [])

        for dep in deps:
            relation = dep.get("relation", "")
            word = dep.get("word", "")

            if relation == "nsubj":
                # Subject is usually agent or experiencer
                if frame_type in ["perception", "cognition"]:
                    roles[SemanticRole.EXPERIENCER] = word
                else:
                    roles[SemanticRole.AGENT] = word

            elif relation == "obj":
                # Object role depends on frame type
                if frame_type == "speech":
                    roles[SemanticRole.THEME] = word
                elif frame_type == "giving":
                    roles[SemanticRole.THEME] = word
                elif frame_type == "perception":
                    roles[SemanticRole.STIMULUS] = word
                else:
                    roles[SemanticRole.PATIENT] = word

            elif relation == "iobj":
                roles[SemanticRole.RECIPIENT] = word

            elif relation == "advmod":
                # Adverbials can indicate various roles
                if self._is_location(word):
                    roles[SemanticRole.LOCATION] = word
                elif self._is_temporal(word):
                    roles[SemanticRole.TIME] = word
                else:
                    roles[SemanticRole.MANNER] = word

        return roles

    def _is_location(self, word: str) -> bool:
        """Check if word is a location indicator."""
        locations = ["where", "here", "there", "heaven", "earth", "temple"]
        return word.lower() in locations

    def _is_temporal(self, word: str) -> bool:
        """Check if word is a temporal indicator."""
        temporals = ["when", "now", "then", "today", "forever", "always"]
        return word.lower() in temporals

    def _determine_frame_domain(
        self,
        text: str,
        frame_type: str
    ) -> Optional[SemanticDomain]:
        """Determine semantic domain of frame."""
        text_lower = text.lower()

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw.lower() in text_lower for kw in keywords):
                return domain

        return None

    def _identify_domains(self, text: str) -> List[SemanticDomain]:
        """Identify all semantic domains present in text."""
        domains: Set[SemanticDomain] = set()
        text_lower = text.lower()

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    domains.add(domain)
                    break

        return list(domains)

    def _label_roles(
        self,
        text: str,
        syntax_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Label semantic roles for all predicates."""
        roles = []
        tokens = self._tokenize(text)

        # Simple heuristic role assignment
        for i, token in enumerate(tokens):
            # Check if this could be a predicate
            for frame_type, verbs in self.FRAME_VERBS.items():
                if token in [v.lower() for v in verbs]:
                    # Assign roles to surrounding words
                    role_assignment = {
                        "predicate": token,
                        "position": i,
                        "roles": []
                    }

                    # Previous word might be subject/agent
                    if i > 0:
                        role_assignment["roles"].append({
                            "word": tokens[i - 1],
                            "role": SemanticRole.AGENT.value
                        })

                    # Following word might be object/patient
                    if i < len(tokens) - 1:
                        role_assignment["roles"].append({
                            "word": tokens[i + 1],
                            "role": SemanticRole.PATIENT.value
                        })

                    roles.append(role_assignment)

        return roles

    def _extract_relationships(
        self,
        text: str,
        frames: List[SemanticFrame]
    ) -> List[Dict[str, Any]]:
        """Extract semantic relationships between entities."""
        relationships = []

        for frame in frames:
            # Extract relationships from frame roles
            if SemanticRole.AGENT in frame.roles and SemanticRole.PATIENT in frame.roles:
                relationships.append({
                    "type": "action",
                    "source": frame.roles[SemanticRole.AGENT],
                    "target": frame.roles[SemanticRole.PATIENT],
                    "predicate": frame.predicate
                })

            if SemanticRole.AGENT in frame.roles and SemanticRole.RECIPIENT in frame.roles:
                relationships.append({
                    "type": "transfer",
                    "source": frame.roles[SemanticRole.AGENT],
                    "target": frame.roles[SemanticRole.RECIPIENT],
                    "predicate": frame.predicate
                })

        return relationships

    def _extract_key_concepts(
        self,
        text: str,
        domains: List[SemanticDomain]
    ) -> List[str]:
        """Extract key theological/semantic concepts."""
        concepts = []
        text_lower = text.lower()

        for domain in domains:
            keywords = self.DOMAIN_KEYWORDS.get(domain, [])
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    concepts.append(keyword)

        return list(set(concepts))

    def _calculate_density(
        self,
        frames: List[SemanticFrame],
        domains: List[SemanticDomain]
    ) -> float:
        """Calculate semantic density score."""
        # Higher density = more semantic content
        frame_score = len(frames) * 0.3
        domain_score = len(domains) * 0.2

        return min(1.0, frame_score + domain_score + 0.3)

    def _calculate_confidence(
        self,
        frames: List[SemanticFrame],
        domains: List[SemanticDomain],
        roles: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score."""
        # Base confidence
        confidence = 0.5

        # Frames boost confidence
        if frames:
            confidence += 0.2

        # Domains boost confidence
        if domains:
            confidence += 0.15

        # Role assignments boost confidence
        if roles:
            confidence += 0.15

        return min(1.0, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "frames" in data and "domains" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus", "morphologos", "syntaktikos"]
