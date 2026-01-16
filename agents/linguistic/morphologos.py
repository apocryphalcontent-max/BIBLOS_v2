"""
BIBLOS v2 - MORPHOLOGOS Agent

Morphological analysis agent for Greek and Hebrew biblical texts.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
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


class PartOfSpeech(Enum):
    """Parts of speech for biblical languages."""
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PRONOUN = "pronoun"
    ARTICLE = "article"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    PARTICLE = "particle"
    INTERJECTION = "interjection"
    NUMERAL = "numeral"


class Case(Enum):
    """Grammatical cases."""
    NOMINATIVE = "nominative"
    GENITIVE = "genitive"
    DATIVE = "dative"
    ACCUSATIVE = "accusative"
    VOCATIVE = "vocative"
    ABSOLUTE = "absolute"  # Hebrew construct


class Number(Enum):
    """Grammatical number."""
    SINGULAR = "singular"
    PLURAL = "plural"
    DUAL = "dual"


class Gender(Enum):
    """Grammatical gender."""
    MASCULINE = "masculine"
    FEMININE = "feminine"
    NEUTER = "neuter"
    COMMON = "common"


class Tense(Enum):
    """Verbal tenses."""
    PRESENT = "present"
    IMPERFECT = "imperfect"
    AORIST = "aorist"
    PERFECT = "perfect"
    PLUPERFECT = "pluperfect"
    FUTURE = "future"
    FUTURE_PERFECT = "future_perfect"


class Voice(Enum):
    """Verbal voices."""
    ACTIVE = "active"
    MIDDLE = "middle"
    PASSIVE = "passive"
    MIDDLE_PASSIVE = "middle_passive"


class Mood(Enum):
    """Verbal moods."""
    INDICATIVE = "indicative"
    SUBJUNCTIVE = "subjunctive"
    OPTATIVE = "optative"
    IMPERATIVE = "imperative"
    INFINITIVE = "infinitive"
    PARTICIPLE = "participle"


@dataclass
class MorphologicalTag:
    """Complete morphological analysis of a word."""
    lemma: str
    pos: PartOfSpeech
    case: Optional[Case] = None
    number: Optional[Number] = None
    gender: Optional[Gender] = None
    tense: Optional[Tense] = None
    voice: Optional[Voice] = None
    mood: Optional[Mood] = None
    person: Optional[int] = None  # 1, 2, 3
    stem: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    root: Optional[str] = None  # For Hebrew

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lemma": self.lemma,
            "pos": self.pos.value if self.pos else None,
            "case": self.case.value if self.case else None,
            "number": self.number.value if self.number else None,
            "gender": self.gender.value if self.gender else None,
            "tense": self.tense.value if self.tense else None,
            "voice": self.voice.value if self.voice else None,
            "mood": self.mood.value if self.mood else None,
            "person": self.person,
            "stem": self.stem,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "root": self.root
        }


class MorphologosAgent(BaseExtractionAgent):
    """
    MORPHOLOGOS - Morphological analysis agent.

    Analyzes word forms in Greek and Hebrew:
    - Part of speech identification
    - Case, number, gender analysis
    - Verbal parsing (tense, voice, mood, person)
    - Lemmatization
    - Root extraction (Hebrew)
    """

    # Greek article forms for detection
    GREEK_ARTICLES = {
        "ὁ", "ἡ", "τό", "οἱ", "αἱ", "τά",
        "τοῦ", "τῆς", "τῶν", "τῷ", "τῇ", "τοῖς", "ταῖς",
        "τόν", "τήν", "τούς", "τάς"
    }

    # Common Greek prepositions
    GREEK_PREPOSITIONS = {
        "ἐν", "εἰς", "ἐκ", "ἐξ", "ἀπό", "πρός", "διά",
        "κατά", "μετά", "περί", "ὑπέρ", "ὑπό", "παρά",
        "ἐπί", "πρό", "σύν", "ἀντί"
    }

    # Hebrew prefixes
    HEBREW_PREFIXES = {
        "ב": "in/with",
        "כ": "like/as",
        "ל": "to/for",
        "מ": "from",
        "ו": "and",
        "ה": "the/interrogative",
        "ש": "that/which"
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="morphologos",
                extraction_type=ExtractionType.MORPHOLOGICAL,
                batch_size=500,
                min_confidence=0.7
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.morphologos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """
        Extract morphological analysis from verse.
        """
        language = self._detect_language(text)
        words = self._tokenize(text)
        analyses = []

        for word in words:
            if language == "greek":
                analysis = self._analyze_greek(word)
            elif language == "hebrew":
                analysis = self._analyze_hebrew(word)
            else:
                analysis = self._analyze_default(word)

            analyses.append({
                "word": word,
                "analysis": analysis.to_dict() if analysis else None
            })

        data = {
            "language": language,
            "word_count": len(words),
            "analyses": analyses,
            "summary": self._summarize_morphology(analyses)
        }

        confidence = self._calculate_confidence(analyses)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _detect_language(self, text: str) -> str:
        """Detect primary language of text."""
        greek_chars = len(re.findall(r'[\u0370-\u03FF\u1F00-\u1FFF]', text))
        hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))

        if greek_chars > hebrew_chars and greek_chars > latin_chars:
            return "greek"
        elif hebrew_chars > greek_chars and hebrew_chars > latin_chars:
            return "hebrew"
        return "english"

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text preserving original forms."""
        pattern = r'[\w\u0370-\u03FF\u1F00-\u1FFF\u0590-\u05FF]+'
        return re.findall(pattern, text)

    def _analyze_greek(self, word: str) -> Optional[MorphologicalTag]:
        """Analyze Greek word morphology."""
        word_lower = word.lower()

        # Check for article
        if word_lower in self.GREEK_ARTICLES:
            return MorphologicalTag(
                lemma="ὁ",
                pos=PartOfSpeech.ARTICLE,
                case=self._infer_greek_case(word_lower),
                number=self._infer_greek_number(word_lower),
                gender=self._infer_greek_gender(word_lower)
            )

        # Check for preposition
        if word_lower in self.GREEK_PREPOSITIONS:
            return MorphologicalTag(
                lemma=word_lower,
                pos=PartOfSpeech.PREPOSITION
            )

        # Analyze verb endings
        if self._is_greek_verb(word_lower):
            return self._parse_greek_verb(word_lower)

        # Default noun/adjective analysis
        return self._parse_greek_nominal(word_lower)

    def _analyze_hebrew(self, word: str) -> Optional[MorphologicalTag]:
        """Analyze Hebrew word morphology."""
        # Extract prefixes
        prefixes = []
        stem = word

        for prefix, meaning in self.HEBREW_PREFIXES.items():
            if stem.startswith(prefix) and len(stem) > 1:
                prefixes.append(prefix)
                stem = stem[1:]

        # Analyze verbal patterns (binyanim)
        if self._is_hebrew_verb(stem):
            return self._parse_hebrew_verb(stem, prefixes)

        # Nominal analysis
        return MorphologicalTag(
            lemma=stem,
            pos=PartOfSpeech.NOUN,
            prefix=",".join(prefixes) if prefixes else None,
            root=self._extract_hebrew_root(stem)
        )

    def _analyze_default(self, word: str) -> MorphologicalTag:
        """Default analysis for unrecognized language."""
        return MorphologicalTag(
            lemma=word.lower(),
            pos=PartOfSpeech.NOUN
        )

    def _is_greek_verb(self, word: str) -> bool:
        """Check if word appears to be a Greek verb."""
        verb_endings = ["ω", "ει", "ομεν", "ετε", "ουσι", "ειν", "ων", "ας", "εν"]
        return any(word.endswith(ending) for ending in verb_endings)

    def _parse_greek_verb(self, word: str) -> MorphologicalTag:
        """Parse Greek verb form."""
        # Simplified parsing - would use full morphological database in production
        tense = Tense.PRESENT
        mood = Mood.INDICATIVE
        voice = Voice.ACTIVE
        person = 3
        number = Number.SINGULAR

        if word.endswith("ομεν") or word.endswith("ετε"):
            number = Number.PLURAL
        if word.endswith("ειν"):
            mood = Mood.INFINITIVE
        if word.endswith("ων") or word.endswith("ας"):
            mood = Mood.PARTICIPLE

        return MorphologicalTag(
            lemma=word,  # Would look up actual lemma
            pos=PartOfSpeech.VERB,
            tense=tense,
            voice=voice,
            mood=mood,
            person=person if mood == Mood.INDICATIVE else None,
            number=number
        )

    def _parse_greek_nominal(self, word: str) -> MorphologicalTag:
        """Parse Greek nominal (noun/adjective) form."""
        case = self._infer_greek_case(word)
        number = self._infer_greek_number(word)
        gender = self._infer_greek_gender(word)

        return MorphologicalTag(
            lemma=word,
            pos=PartOfSpeech.NOUN,
            case=case,
            number=number,
            gender=gender
        )

    def _infer_greek_case(self, word: str) -> Optional[Case]:
        """Infer case from Greek word ending."""
        if word.endswith(("ος", "ον", "α", "η")):
            return Case.NOMINATIVE
        elif word.endswith(("ου", "ης", "ων")):
            return Case.GENITIVE
        elif word.endswith(("ῳ", "ῃ", "οις", "αις")):
            return Case.DATIVE
        elif word.endswith(("ον", "ην", "ους", "ας")):
            return Case.ACCUSATIVE
        return None

    def _infer_greek_number(self, word: str) -> Optional[Number]:
        """Infer number from Greek word ending."""
        plural_endings = ("οι", "αι", "α", "ων", "οις", "αις", "ους", "ας")
        if word.endswith(plural_endings):
            return Number.PLURAL
        return Number.SINGULAR

    def _infer_greek_gender(self, word: str) -> Optional[Gender]:
        """Infer gender from Greek word ending."""
        if word.endswith(("ος", "ου", "ον")):
            return Gender.MASCULINE
        elif word.endswith(("η", "α", "ης")):
            return Gender.FEMININE
        elif word.endswith("ον"):
            return Gender.NEUTER
        return None

    def _is_hebrew_verb(self, word: str) -> bool:
        """Check if Hebrew word appears to be a verb."""
        # Hebrew verbs often have specific patterns
        return len(word) >= 3 and any(
            word.startswith(prefix) for prefix in ["י", "ת", "א", "נ"]
        )

    def _parse_hebrew_verb(
        self,
        word: str,
        prefixes: List[str]
    ) -> MorphologicalTag:
        """Parse Hebrew verb form."""
        root = self._extract_hebrew_root(word)

        return MorphologicalTag(
            lemma=word,
            pos=PartOfSpeech.VERB,
            root=root,
            prefix=",".join(prefixes) if prefixes else None
        )

    def _extract_hebrew_root(self, word: str) -> Optional[str]:
        """Extract three-letter root from Hebrew word."""
        # Remove vowel points and prefixes/suffixes
        consonants = re.sub(r'[\u05B0-\u05BD\u05BF-\u05C7]', '', word)
        if len(consonants) >= 3:
            return consonants[:3]
        return None

    def _summarize_morphology(
        self,
        analyses: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Summarize morphological analysis."""
        pos_counts: Dict[str, int] = {}
        for item in analyses:
            if item["analysis"] and item["analysis"].get("pos"):
                pos = item["analysis"]["pos"]
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
        return pos_counts

    def _calculate_confidence(
        self,
        analyses: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score."""
        if not analyses:
            return 0.0

        analyzed = sum(1 for a in analyses if a["analysis"] is not None)
        return analyzed / max(1, len(analyses))

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "analyses" in data and "language" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus"]  # Depends on text tokenization
