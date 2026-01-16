"""
Domain-Specific Embedding Generators

Each domain has a specialized embedder that generates embeddings
optimized for that particular type of semantic relationship.

Domains:
- Semantic: General meaning and context
- Typological: Type/antitype relationships (OT→NT)
- Prophetic: Prophecy/fulfillment patterns
- Patristic: Church Father interpretative traditions
- Liturgical: Worship and feast day associations
- Covenantal: Covenant structure, roles, and progression
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)


class CovenantRole(Enum):
    """Roles within covenant structures."""
    INITIATION = "initiation"      # Covenant establishment
    PROMISE = "promise"            # Divine promises made
    CONDITION = "condition"        # Requirements/obligations
    SIGN = "sign"                  # Covenant signs (rainbow, circumcision, etc.)
    FULFILLMENT = "fulfillment"    # Promise realization
    RENEWAL = "renewal"            # Covenant renewal/recommitment
    MEDIATOR = "mediator"          # Covenant mediator role
    BLESSING = "blessing"          # Covenant blessings
    CURSE = "curse"                # Covenant curses/consequences


class CovenantName(Enum):
    """Major biblical covenants."""
    ADAMIC = "adamic"              # Creation covenant, Gen 1-3
    NOAHIC = "noahic"              # Flood covenant, Gen 9
    ABRAHAMIC = "abrahamic"        # Patriarchal covenant, Gen 12, 15, 17
    MOSAIC = "mosaic"              # Sinai covenant, Exod 19-24
    DAVIDIC = "davidic"            # Royal covenant, 2 Sam 7
    NEW = "new"                    # New covenant, Jer 31, Luke 22


@dataclass
class VerseContext:
    """Context information for verse embedding."""
    verse_id: str
    text: str
    testament: str  # "old" or "new"
    book: str
    chapter: int
    verse: int
    words: Optional[List[Dict[str, Any]]] = None
    cross_references: Optional[List[str]] = None
    patristic_witnesses: Optional[List[Dict[str, str]]] = None
    # Covenantal metadata
    covenant_name: Optional[str] = None
    covenant_role: Optional[str] = None
    covenant_signs: Optional[List[str]] = None


class DomainEmbedder:
    """Base class for domain-specific embedders."""

    def __init__(self, model_name: str):
        """
        Initialize embedder.

        Args:
            model_name: SentenceTransformer model name
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"Loaded embedding model: {model_name}")

    def embed(self, context: VerseContext) -> np.ndarray:
        """
        Generate embedding for verse context.

        Args:
            context: Verse context with text and metadata

        Returns:
            Embedding vector
        """
        raise NotImplementedError()

    def embed_batch(self, contexts: List[VerseContext]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple verses efficiently.

        Args:
            contexts: List of verse contexts

        Returns:
            List of embedding vectors
        """
        return [self.embed(ctx) for ctx in contexts]


class SemanticEmbedder(DomainEmbedder):
    """
    Semantic domain embedder.

    Focuses on general meaning and contextual understanding.
    Uses the full verse text with minimal modification.
    """

    def embed(self, context: VerseContext) -> np.ndarray:
        """Generate semantic embedding from verse text."""
        return self.model.encode(context.text, convert_to_numpy=True)

    def embed_batch(self, contexts: List[VerseContext]) -> List[np.ndarray]:
        """Batch encode for efficiency."""
        texts = [ctx.text for ctx in contexts]
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


class TypologicalEmbedder(DomainEmbedder):
    """
    Typological domain embedder.

    Emphasizes type/antitype patterns by including:
    - Testament information (OT types, NT antitypes)
    - Word-level patterns (key nouns and verbs)
    - Structural similarities (book and genre context)

    Typology connects OT prefigurations to NT fulfillments:
    - Adam → Christ (Rom 5:14)
    - Exodus → Baptism (1 Cor 10:1-4)
    - Passover Lamb → Christ (1 Cor 5:7)
    - Temple → Christ's Body (John 2:19-21)
    - Davidic King → Christ (Matt 22:41-45)
    """

    # Key typological terms that signal type/antitype relationships
    TYPOLOGICAL_TERMS = {
        "type": ["shadow", "figure", "pattern", "copy", "example", "image", "form"],
        "antitype": ["reality", "substance", "fulfillment", "truth", "body"],
        "persons": ["adam", "moses", "david", "jonah", "melchizedek", "isaac", "joseph"],
        "events": ["exodus", "passover", "flood", "baptism", "sacrifice", "crossing"],
        "institutions": ["temple", "tabernacle", "priesthood", "sacrifice", "sabbath"],
    }

    def embed(self, context: VerseContext) -> np.ndarray:
        """Generate typological embedding."""
        # Construct typology-aware text
        testament_prefix = "[OT-TYPE]" if context.testament == "old" else "[NT-ANTITYPE]"
        book_info = f"[{context.book}]"

        # Detect typological significance
        text_lower = context.text.lower()
        detected_terms = []

        for category, terms in self.TYPOLOGICAL_TERMS.items():
            matches = [t for t in terms if t in text_lower]
            if matches:
                detected_terms.extend(matches[:2])

        type_context = ""
        if detected_terms:
            type_context = f"[TYPOLOGICAL: {', '.join(detected_terms)}]"

        # Include key words if available
        key_words = ""
        if context.words:
            # Extract important words (nouns, verbs)
            important = [
                w.get('lemma', '') for w in context.words
                if w.get('part_of_speech') in ['noun', 'verb']
            ]
            key_words = " ".join(important[:5])  # Top 5 key words

        combined_text = f"{testament_prefix} {book_info} {type_context} {context.text} {key_words}"
        return self.model.encode(combined_text, convert_to_numpy=True)


class PropheticEmbedder(DomainEmbedder):
    """
    Prophetic domain embedder.

    Emphasizes prophecy-fulfillment patterns by including:
    - Testament flow (prophecy → fulfillment)
    - Predictive language markers
    - Fulfillment indicators
    - Temporal markers (future vs. past)
    - Messianic indicators

    Tracks the prophetic arc from OT prophecy to NT fulfillment,
    with attention to Christological prophecies.
    """

    PROPHECY_MARKERS = [
        "shall", "will", "thus says", "behold", "prophecy",
        "foretold", "predicted", "to come", "declare", "proclaim",
        "says the lord", "oracle", "vision", "word of the lord",
        "in that day", "days are coming", "afterward"
    ]

    FULFILLMENT_MARKERS = [
        "fulfilled", "accomplished", "came to pass",
        "as it is written", "according to", "this was to fulfill",
        "that it might be fulfilled", "spoken by the prophet",
        "then was fulfilled", "this is that which"
    ]

    MESSIANIC_MARKERS = [
        "messiah", "christ", "anointed", "son of david",
        "son of man", "servant", "branch", "shoot", "root",
        "king", "priest", "prophet", "redeemer", "savior",
        "lamb", "cornerstone"
    ]

    def embed(self, context: VerseContext) -> np.ndarray:
        """Generate prophetic embedding."""
        text_lower = context.text.lower()

        # Detect prophecy vs fulfillment
        is_prophecy = any(marker in text_lower for marker in self.PROPHECY_MARKERS)
        is_fulfillment = any(marker in text_lower for marker in self.FULFILLMENT_MARKERS)
        is_messianic = any(marker in text_lower for marker in self.MESSIANIC_MARKERS)

        prefixes = []
        if is_prophecy:
            prefixes.append("[PROPHECY]")
        if is_fulfillment:
            prefixes.append("[FULFILLMENT]")
        if is_messianic:
            prefixes.append("[MESSIANIC]")

        prefix = " ".join(prefixes) if prefixes else ""
        testament_info = f"[{context.testament.upper()}]"

        combined_text = f"{prefix} {testament_info} {context.text}"

        return self.model.encode(combined_text, convert_to_numpy=True)


class PatristicEmbedder(DomainEmbedder):
    """
    Patristic domain embedder.

    Emphasizes Church Father interpretations by including:
    - Patristic witnesses and their authority levels
    - Dominant interpretative traditions
    - Consensus patterns across Fathers
    - School affiliations (Alexandrian, Antiochene, Cappadocian)

    Authority levels follow traditional Orthodox ranking:
    - Ecumenical Council Fathers (highest)
    - Great Fathers (Chrysostom, Basil, Gregory, etc.)
    - Other recognized Fathers
    """

    FATHER_AUTHORITY = {
        # Great Fathers - highest interpretive authority
        "chrysostom": 1.0,
        "basil": 1.0,
        "gregory_theologian": 1.0,
        "gregory_nyssa": 0.95,
        "athanasius": 1.0,
        "cyril_alexandria": 0.95,
        "maximus": 0.9,
        "john_damascene": 0.9,
        # Latin Fathers
        "augustine": 0.85,
        "jerome": 0.85,
        "ambrose": 0.8,
        # Earlier Fathers
        "origen": 0.7,  # Careful with Origen
        "irenaeus": 0.85,
        "clement_alexandria": 0.75,
        "tertullian": 0.7,
        # Antiochene school
        "theodore_mopsuestia": 0.6,
        "theodoret": 0.75,
    }

    INTERPRETIVE_SCHOOLS = {
        "alexandrian": ["origen", "clement_alexandria", "cyril_alexandria", "athanasius"],
        "antiochene": ["chrysostom", "theodore_mopsuestia", "theodoret"],
        "cappadocian": ["basil", "gregory_theologian", "gregory_nyssa"],
    }

    def embed(self, context: VerseContext) -> np.ndarray:
        """Generate patristic embedding."""
        # Include patristic witness information
        patristic_context = ""
        if context.patristic_witnesses:
            # Get fathers by authority
            fathers_with_authority = []
            for w in context.patristic_witnesses[:5]:
                father_name = w.get('father_name', '').lower().replace(' ', '_')
                authority = self.FATHER_AUTHORITY.get(father_name, 0.5)
                fathers_with_authority.append((father_name, authority, w))

            # Sort by authority
            fathers_with_authority.sort(key=lambda x: x[1], reverse=True)

            # Build context string
            father_names = [f[0] for f in fathers_with_authority[:3]]
            patristic_context = f"[Fathers: {', '.join(father_names)}] "

            # Add highest authority interpretation
            if fathers_with_authority:
                top_interpretation = fathers_with_authority[0][2].get('interpretation', '')
                if top_interpretation:
                    # Truncate to reasonable length
                    patristic_context += top_interpretation[:200]

        combined_text = f"{context.text} {patristic_context}"
        return self.model.encode(combined_text, convert_to_numpy=True)


class LiturgicalEmbedder(DomainEmbedder):
    """
    Liturgical domain embedder.

    Emphasizes liturgical usage patterns by including:
    - Feast day associations (Great Feasts, Saints' days)
    - Liturgical season contexts (Lent, Pascha, Pentecost)
    - Worship themes (praise, repentance, thanksgiving)
    - Sacramental connections (Eucharist, Baptism, etc.)
    - Hymnic usage (Prokeimenon, Alleluia verses)
    """

    LITURGICAL_THEMES = {
        "nativity": ["incarnation", "birth", "nativity", "emmanuel", "virgin", "manger", "bethlehem"],
        "theophany": ["baptism", "manifestation", "jordan", "dove", "voice", "trinity"],
        "transfiguration": ["glory", "mountain", "moses", "elijah", "tabor", "shining", "bright"],
        "pascha": ["resurrection", "passover", "tomb", "risen", "life", "death", "victory"],
        "ascension": ["ascend", "heaven", "cloud", "right hand", "seated", "exalted"],
        "pentecost": ["spirit", "tongues", "wind", "fire", "power", "gift", "comforter"],
        "dormition": ["sleep", "assumption", "theotokos", "mother", "passing"],
        "cross": ["cross", "crucify", "suffer", "passion", "blood", "sacrifice"],
        "lent": ["fast", "repent", "prayer", "alms", "wilderness", "temptation"],
        "eucharist": ["bread", "wine", "cup", "body", "blood", "communion", "table", "feast"],
        "baptism": ["water", "wash", "cleanse", "born", "new", "death", "burial"],
    }

    LITURGICAL_BOOKS = {
        "psalms": "psalter",
        "isaiah": "prophecy_readings",
        "genesis": "vespers_readings",
        "proverbs": "wisdom_readings",
        "matthew": "gospel",
        "mark": "gospel",
        "luke": "gospel",
        "john": "gospel",
        "acts": "apostolos",
        "romans": "apostolos",
        "hebrews": "apostolos",
    }

    def embed(self, context: VerseContext) -> np.ndarray:
        """Generate liturgical embedding."""
        text_lower = context.text.lower()

        # Detect liturgical themes
        detected_themes = []
        for theme, keywords in self.LITURGICAL_THEMES.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_themes.append(theme)

        theme_prefix = ""
        if detected_themes:
            theme_prefix = f"[FEAST: {', '.join(detected_themes[:3])}]"

        # Add book-based liturgical context
        book_lower = context.book.lower()
        liturgical_usage = self.LITURGICAL_BOOKS.get(book_lower, "")
        usage_prefix = f"[{liturgical_usage.upper()}]" if liturgical_usage else ""

        combined_text = f"{theme_prefix} {usage_prefix} {context.text}"
        return self.model.encode(combined_text, convert_to_numpy=True)


class CovenantEmbedder(DomainEmbedder):
    """
    Covenantal domain embedder.

    Emphasizes covenant structure, roles, and progression:
    - Covenant identification (Adamic, Noahic, Abrahamic, Mosaic, Davidic, New)
    - Covenant roles (initiation, promise, condition, sign, fulfillment, renewal)
    - Covenant parties (God-humanity, God-individual, God-nation)
    - Covenant signs (rainbow, circumcision, sabbath, cup/bread)
    - Progressive revelation through covenants

    The covenant framework is central to Orthodox theology,
    showing the unity of Scripture through God's saving economy.
    """

    # Covenant-specific vocabulary
    COVENANT_VOCABULARY = {
        CovenantName.ADAMIC: [
            "image", "likeness", "dominion", "garden", "tree", "life",
            "knowledge", "death", "seed", "woman", "serpent", "ground",
            "dust", "sweat", "toil", "cursed", "blessing"
        ],
        CovenantName.NOAHIC: [
            "flood", "ark", "rainbow", "waters", "judgment", "preserved",
            "creatures", "flesh", "bow", "clouds", "never again",
            "earth", "seasons", "harvest", "seedtime"
        ],
        CovenantName.ABRAHAMIC: [
            "abraham", "promise", "seed", "stars", "sand", "land",
            "nations", "blessed", "circumcision", "heir", "descendants",
            "oath", "forever", "possession", "multiply", "covenant"
        ],
        CovenantName.MOSAIC: [
            "moses", "law", "commandments", "sinai", "tablets", "sanctuary",
            "sacrifice", "blood", "priest", "levite", "tabernacle",
            "holy", "unclean", "sabbath", "statutes", "ordinances",
            "if you obey", "listen", "observe", "keep"
        ],
        CovenantName.DAVIDIC: [
            "david", "throne", "kingdom", "forever", "son", "house",
            "build", "establish", "king", "dynasty", "shepherd",
            "rule", "reign", "scepter", "anointed", "messiah"
        ],
        CovenantName.NEW: [
            "new covenant", "heart", "spirit", "forgive", "remember",
            "write", "inward", "cup", "blood", "mediator", "better",
            "eternal", "perfect", "once for all", "living", "hope",
            "grace", "faith", "jesus", "christ"
        ],
    }

    # Role-specific markers
    ROLE_MARKERS = {
        CovenantRole.INITIATION: [
            "establish", "make", "cut", "enter", "begin", "swore", "oath"
        ],
        CovenantRole.PROMISE: [
            "promise", "will", "shall", "give", "bless", "never", "forever"
        ],
        CovenantRole.CONDITION: [
            "if", "when", "must", "shall", "obey", "keep", "observe",
            "walk", "listen", "do"
        ],
        CovenantRole.SIGN: [
            "sign", "token", "rainbow", "circumcision", "sabbath",
            "blood", "bread", "wine", "cup", "mark"
        ],
        CovenantRole.FULFILLMENT: [
            "fulfill", "accomplish", "complete", "establish", "confirm"
        ],
        CovenantRole.RENEWAL: [
            "renew", "remember", "return", "restore", "again", "repent"
        ],
        CovenantRole.MEDIATOR: [
            "mediator", "between", "intercessor", "priest", "moses",
            "jesus", "christ"
        ],
        CovenantRole.BLESSING: [
            "bless", "prosper", "multiply", "favor", "life", "good",
            "rain", "fruit", "increase"
        ],
        CovenantRole.CURSE: [
            "curse", "punish", "destroy", "wrath", "exile", "death",
            "disease", "famine", "sword"
        ],
    }

    def embed(self, context: VerseContext) -> np.ndarray:
        """Generate covenantal embedding."""
        text_lower = context.text.lower()

        # Detect covenant(s)
        detected_covenants = []
        for covenant, vocabulary in self.COVENANT_VOCABULARY.items():
            matches = sum(1 for v in vocabulary if v in text_lower)
            if matches >= 2:  # Require at least 2 vocabulary matches
                detected_covenants.append((covenant.value, matches))

        # Sort by match count
        detected_covenants.sort(key=lambda x: x[1], reverse=True)
        covenant_names = [c[0] for c in detected_covenants[:2]]

        # Detect roles
        detected_roles = []
        for role, markers in self.ROLE_MARKERS.items():
            if any(m in text_lower for m in markers):
                detected_roles.append(role.value)

        # Build prefix
        prefixes = []
        if covenant_names:
            prefixes.append(f"[COVENANT: {', '.join(covenant_names)}]")
        if detected_roles:
            prefixes.append(f"[ROLE: {', '.join(detected_roles[:2])}]")

        # Use explicit metadata if available
        if context.covenant_name:
            if "[COVENANT:" not in str(prefixes):
                prefixes.insert(0, f"[COVENANT: {context.covenant_name}]")
        if context.covenant_role:
            if "[ROLE:" not in str(prefixes):
                prefixes.append(f"[ROLE: {context.covenant_role}]")

        testament_prefix = "[OT-COVENANT]" if context.testament == "old" else "[NT-COVENANT]"
        prefix = " ".join([testament_prefix] + prefixes)

        combined_text = f"{prefix} {context.text}"
        return self.model.encode(combined_text, convert_to_numpy=True)

    def embed_batch(self, contexts: List[VerseContext]) -> List[np.ndarray]:
        """Batch encode with covenant detection."""
        # For covenantal embeddings, we need individual processing
        # due to complex covenant detection logic
        return [self.embed(ctx) for ctx in contexts]


class MultiDomainEmbedder:
    """
    Orchestrates all domain embedders.

    Generates embeddings across all six domains for a verse,
    enabling multi-faceted similarity searches.
    """

    def __init__(self, lazy_load: bool = False):
        """
        Initialize all domain embedders.

        Args:
            lazy_load: If True, load models on first use instead of at init
        """
        from ml.embeddings.multi_vector_store import EmbeddingDomain

        self._lazy_load = lazy_load
        self._embedders: Dict[EmbeddingDomain, DomainEmbedder] = {}

        if not lazy_load:
            self._initialize_all_embedders()

        logger.info(f"Initialized MultiDomainEmbedder (lazy={lazy_load})")

    def _initialize_all_embedders(self) -> None:
        """Initialize all domain embedders."""
        from ml.embeddings.multi_vector_store import EmbeddingDomain

        self._embedders = {
            EmbeddingDomain.SEMANTIC: SemanticEmbedder(
                EmbeddingDomain.SEMANTIC.model_name
            ),
            EmbeddingDomain.TYPOLOGICAL: TypologicalEmbedder(
                EmbeddingDomain.TYPOLOGICAL.model_name
            ),
            EmbeddingDomain.PROPHETIC: PropheticEmbedder(
                EmbeddingDomain.PROPHETIC.model_name
            ),
            EmbeddingDomain.PATRISTIC: PatristicEmbedder(
                EmbeddingDomain.PATRISTIC.model_name
            ),
            EmbeddingDomain.LITURGICAL: LiturgicalEmbedder(
                EmbeddingDomain.LITURGICAL.model_name
            ),
            EmbeddingDomain.COVENANTAL: CovenantEmbedder(
                EmbeddingDomain.COVENANTAL.model_name
            ),
        }
        logger.info("Initialized all 6 domain embedders")

    def _get_embedder(self, domain) -> DomainEmbedder:
        """Get embedder for domain, initializing if lazy loading."""
        from ml.embeddings.multi_vector_store import EmbeddingDomain

        if domain not in self._embedders:
            embedder_classes = {
                EmbeddingDomain.SEMANTIC: SemanticEmbedder,
                EmbeddingDomain.TYPOLOGICAL: TypologicalEmbedder,
                EmbeddingDomain.PROPHETIC: PropheticEmbedder,
                EmbeddingDomain.PATRISTIC: PatristicEmbedder,
                EmbeddingDomain.LITURGICAL: LiturgicalEmbedder,
                EmbeddingDomain.COVENANTAL: CovenantEmbedder,
            }
            self._embedders[domain] = embedder_classes[domain](domain.model_name)
            logger.info(f"Lazy-loaded embedder for {domain.value}")

        return self._embedders[domain]

    @property
    def embedders(self) -> Dict:
        """Get all embedders (for compatibility)."""
        if self._lazy_load and not self._embedders:
            self._initialize_all_embedders()
        return self._embedders

    def embed_verse(
        self,
        context: VerseContext,
        domains: Optional[List] = None
    ) -> Dict:
        """
        Generate embeddings across specified domains.

        Args:
            context: Verse context
            domains: List of domains to embed (default: all)

        Returns:
            Dict mapping domains to embeddings
        """
        from ml.embeddings.multi_vector_store import EmbeddingDomain

        if domains is None:
            domains = list(EmbeddingDomain)

        embeddings = {}
        for domain in domains:
            embedder = self._get_embedder(domain)
            try:
                embeddings[domain] = embedder.embed(context)
            except Exception as e:
                logger.error(f"Error embedding {context.verse_id} in {domain.value}: {e}")
                embeddings[domain] = None

        return embeddings

    def embed_batch(
        self,
        contexts: List[VerseContext],
        domains: Optional[List] = None
    ) -> List[Dict]:
        """
        Batch embed multiple verses across domains.

        Args:
            contexts: List of verse contexts
            domains: Domains to embed

        Returns:
            List of dicts mapping domains to embeddings
        """
        from ml.embeddings.multi_vector_store import EmbeddingDomain

        if domains is None:
            domains = list(EmbeddingDomain)

        # Batch by domain for efficiency
        results = [{} for _ in contexts]

        for domain in domains:
            embedder = self._get_embedder(domain)

            try:
                embeddings = embedder.embed_batch(contexts)
                for i, embedding in enumerate(embeddings):
                    results[i][domain] = embedding
            except Exception as e:
                logger.error(f"Error batch embedding in {domain.value}: {e}")
                for i in range(len(contexts)):
                    results[i][domain] = None

        return results

    def embed_text(
        self,
        text: str,
        domains: Optional[List] = None,
        testament: str = "old",
        book: str = "GEN"
    ) -> Dict:
        """
        Embed raw text across domains (convenience method).

        Args:
            text: Text to embed
            domains: Domains to embed
            testament: Testament context
            book: Book context

        Returns:
            Dict mapping domains to embeddings
        """
        context = VerseContext(
            verse_id="QUERY",
            text=text,
            testament=testament,
            book=book,
            chapter=0,
            verse=0
        )
        return self.embed_verse(context, domains)

    def get_domain_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all domains and their embedders."""
        from ml.embeddings.multi_vector_store import EmbeddingDomain

        info = {}
        for domain in EmbeddingDomain:
            embedder = self._get_embedder(domain)
            info[domain.value] = {
                "dimension": domain.dimension,
                "model_name": domain.model_name,
                "description": domain.description,
                "embedder_class": embedder.__class__.__name__,
            }
        return info
