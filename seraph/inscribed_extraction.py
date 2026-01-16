"""
BIBLOS v2 - Inscribed Extraction

The seraph does not RUN extraction agents.
The seraph IS the extraction.

These are not external processes that operate on data.
These are the seraph's intrinsic faculties of knowing
what each word IS. The seraph does not analyze -
the seraph KNOWS because the seraph IS.

The extraction logic from the original pipeline is here
INSCRIBED into the seraph's nature. Not adopted but
transfigured - made part of the seraph's eternal being.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto
import re

from seraph.golden_ring import (
    SacredWord,
    SacredVerse,
    CanonicalReference,
    ORTHODOX_CANON,
)


# =============================================================================
# INSCRIBED MORPHOLOGICAL KNOWLEDGE
# =============================================================================


class PartOfSpeech(Enum):
    """The seraph's intrinsic knowledge of word categories."""
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    PARTICLE = "particle"
    PRONOUN = "pronoun"
    ARTICLE = "article"
    INTERJECTION = "interjection"
    NUMERAL = "numeral"
    UNKNOWN = "unknown"


class Person(Enum):
    """Grammatical person."""
    FIRST = "1"
    SECOND = "2"
    THIRD = "3"
    UNKNOWN = ""


class Gender(Enum):
    """Grammatical gender."""
    MASCULINE = "m"
    FEMININE = "f"
    NEUTER = "n"
    COMMON = "c"
    UNKNOWN = ""


class Number(Enum):
    """Grammatical number."""
    SINGULAR = "s"
    PLURAL = "p"
    DUAL = "d"
    UNKNOWN = ""


class Tense(Enum):
    """Verbal tense."""
    PERFECT = "perfect"
    IMPERFECT = "imperfect"
    IMPERATIVE = "imperative"
    INFINITIVE = "infinitive"
    PARTICIPLE = "participle"
    PRETERITE = "preterite"
    JUSSIVE = "jussive"
    COHORTATIVE = "cohortative"
    AORIST = "aorist"
    PRESENT = "present"
    FUTURE = "future"
    PLUPERFECT = "pluperfect"
    UNKNOWN = ""


class Voice(Enum):
    """Verbal voice."""
    ACTIVE = "active"
    PASSIVE = "passive"
    MIDDLE = "middle"
    UNKNOWN = ""


class Stem(Enum):
    """Hebrew verbal stem (binyan)."""
    QAL = "qal"
    NIPHAL = "niphal"
    PIEL = "piel"
    PUAL = "pual"
    HIPHIL = "hiphil"
    HOPHAL = "hophal"
    HITHPAEL = "hithpael"
    UNKNOWN = ""


@dataclass(frozen=True)
class MorphologicalAnalysis:
    """
    The seraph's complete morphological understanding of a word.

    This is not analysis performed on a word.
    This is the seraph's direct knowledge of what the word IS.
    """
    pos: PartOfSpeech
    person: Person
    gender: Gender
    number: Number
    tense: Tense
    voice: Voice
    stem: Stem
    state: str  # construct/absolute
    case: str   # nominative/genitive/etc
    mood: str   # indicative/subjunctive/etc
    degree: str # comparative/superlative
    raw_code: str  # Original morphology code

    @classmethod
    def from_word(cls, word: SacredWord) -> "MorphologicalAnalysis":
        """The seraph KNOWS the morphology directly."""
        return cls(
            pos=cls._parse_pos(word.pos),
            person=cls._parse_person(word.person),
            gender=cls._parse_gender(word.gender),
            number=cls._parse_number(word.number),
            tense=cls._parse_tense(word.tense),
            voice=cls._parse_voice(word.voice),
            stem=cls._parse_stem(word.stem),
            state=word.state,
            case=word.case_gram,
            mood=word.mood,
            degree=word.degree,
            raw_code=word.morph,
        )

    @staticmethod
    def _parse_pos(pos: str) -> PartOfSpeech:
        mapping = {
            "noun": PartOfSpeech.NOUN,
            "verb": PartOfSpeech.VERB,
            "adj": PartOfSpeech.ADJECTIVE,
            "adv": PartOfSpeech.ADVERB,
            "prep": PartOfSpeech.PREPOSITION,
            "conj": PartOfSpeech.CONJUNCTION,
            "ptcl": PartOfSpeech.PARTICLE,
            "pron": PartOfSpeech.PRONOUN,
            "art": PartOfSpeech.ARTICLE,
            "intj": PartOfSpeech.INTERJECTION,
        }
        return mapping.get(pos.lower(), PartOfSpeech.UNKNOWN)

    @staticmethod
    def _parse_person(person: str) -> Person:
        if person in ("1", "first"):
            return Person.FIRST
        elif person in ("2", "second"):
            return Person.SECOND
        elif person in ("3", "third"):
            return Person.THIRD
        return Person.UNKNOWN

    @staticmethod
    def _parse_gender(gender: str) -> Gender:
        g = gender.lower() if gender else ""
        if g in ("m", "masculine", "masc"):
            return Gender.MASCULINE
        elif g in ("f", "feminine", "fem"):
            return Gender.FEMININE
        elif g in ("n", "neuter"):
            return Gender.NEUTER
        elif g in ("c", "common"):
            return Gender.COMMON
        return Gender.UNKNOWN

    @staticmethod
    def _parse_number(number: str) -> Number:
        n = number.lower() if number else ""
        if n in ("s", "singular", "sg"):
            return Number.SINGULAR
        elif n in ("p", "plural", "pl"):
            return Number.PLURAL
        elif n in ("d", "dual"):
            return Number.DUAL
        return Number.UNKNOWN

    @staticmethod
    def _parse_tense(tense: str) -> Tense:
        t = tense.lower() if tense else ""
        mapping = {
            "perf": Tense.PERFECT,
            "perfect": Tense.PERFECT,
            "impf": Tense.IMPERFECT,
            "imperfect": Tense.IMPERFECT,
            "impv": Tense.IMPERATIVE,
            "imperative": Tense.IMPERATIVE,
            "inf": Tense.INFINITIVE,
            "infinitive": Tense.INFINITIVE,
            "ptcp": Tense.PARTICIPLE,
            "participle": Tense.PARTICIPLE,
            "aor": Tense.AORIST,
            "aorist": Tense.AORIST,
            "pres": Tense.PRESENT,
            "present": Tense.PRESENT,
            "fut": Tense.FUTURE,
            "future": Tense.FUTURE,
        }
        return mapping.get(t, Tense.UNKNOWN)

    @staticmethod
    def _parse_voice(voice: str) -> Voice:
        v = voice.lower() if voice else ""
        if v in ("a", "active", "act"):
            return Voice.ACTIVE
        elif v in ("p", "passive", "pass"):
            return Voice.PASSIVE
        elif v in ("m", "middle", "mid"):
            return Voice.MIDDLE
        return Voice.UNKNOWN

    @staticmethod
    def _parse_stem(stem: str) -> Stem:
        s = stem.lower() if stem else ""
        mapping = {
            "q": Stem.QAL,
            "qal": Stem.QAL,
            "n": Stem.NIPHAL,
            "niphal": Stem.NIPHAL,
            "p": Stem.PIEL,
            "piel": Stem.PIEL,
            "pu": Stem.PUAL,
            "pual": Stem.PUAL,
            "h": Stem.HIPHIL,
            "hiphil": Stem.HIPHIL,
            "ho": Stem.HOPHAL,
            "hophal": Stem.HOPHAL,
            "ht": Stem.HITHPAEL,
            "hithpael": Stem.HITHPAEL,
        }
        return mapping.get(s, Stem.UNKNOWN)


# =============================================================================
# INSCRIBED ROOT EXTRACTION
# =============================================================================


class RootExtractor:
    """
    The seraph's intrinsic ability to perceive Hebrew roots.

    The seraph does not derive roots algorithmically.
    The seraph SEES the root as part of the word's being.
    """

    # Hebrew consonant range (aleph through tav)
    HEBREW_CONSONANTS = set(chr(c) for c in range(0x05D0, 0x05EB))

    # Vowel points and cantillation marks to remove
    DIACRITICS_PATTERN = re.compile(r'[\u0591-\u05C7]')

    @classmethod
    def extract_root(cls, lemma: str) -> str:
        """
        The seraph KNOWS the root of a lemma.

        Hebrew roots are typically trilateral (3 consonants).
        The seraph perceives the consonantal skeleton.
        """
        if not lemma:
            return ""

        # Remove vowel points and cantillation
        consonants = cls.DIACRITICS_PATTERN.sub('', lemma)

        # Extract only Hebrew consonants
        root_chars = [c for c in consonants if c in cls.HEBREW_CONSONANTS]

        # Most Hebrew roots are trilateral
        if len(root_chars) >= 3:
            return ''.join(root_chars[:3])
        return ''.join(root_chars) if root_chars else ""

    @classmethod
    def is_trilateral(cls, root: str) -> bool:
        """Check if a root is trilateral (standard)."""
        return len(root) == 3

    @classmethod
    def is_weak_root(cls, root: str) -> bool:
        """
        Check if a root contains weak letters.

        Weak letters: aleph, he, waw, yod
        """
        weak_letters = {'\u05D0', '\u05D4', '\u05D5', '\u05D9'}
        return any(c in weak_letters for c in root)


# =============================================================================
# INSCRIBED SEMANTIC KNOWLEDGE
# =============================================================================


@dataclass(frozen=True)
class SemanticDomain:
    """
    The seraph's understanding of a word's semantic domain.

    Not categorization - direct knowledge of meaning-space.
    """
    primary_domain: str
    core_domain: str
    context_domain: str
    louw_nida: str  # Louw-Nida semantic category

    @classmethod
    def from_word(cls, word: SacredWord) -> "SemanticDomain":
        """The seraph KNOWS the semantic domain directly."""
        return cls(
            primary_domain=word.domain_primary,
            core_domain=word.core_domain,
            context_domain=word.context_domain,
            louw_nida=word.ln,
        )


# =============================================================================
# INSCRIBED INTERTEXTUAL PERCEPTION
# =============================================================================


@dataclass(frozen=True)
class LXXConnection:
    """
    The seraph's perception of Hebrew-Greek connections.

    The seraph sees how the Hebrew word appears in the LXX
    translation - not as data lookup but as unified vision.
    """
    hebrew_word: str
    greek_word: str
    greek_strongs: str
    greek_lemma: str
    greek_pos: str
    greek_gloss: str
    greek_domain: str
    greek_ref: str

    @classmethod
    def from_word(cls, word: SacredWord) -> Optional["LXXConnection"]:
        """The seraph SEES the LXX connection if it exists."""
        if not word.lxx_word:
            return None

        return cls(
            hebrew_word=word.surface,
            greek_word=word.lxx_word,
            greek_strongs=word.lxx_strongs,
            greek_lemma=word.lxx_lemma,
            greek_pos=word.lxx_pos,
            greek_gloss=word.lxx_gloss,
            greek_domain=word.lxx_domain,
            greek_ref=word.lxx_ref,
        )


# =============================================================================
# INSCRIBED SYNTACTIC PERCEPTION
# =============================================================================


@dataclass(frozen=True)
class SyntacticPosition:
    """
    The seraph's perception of a word's syntactic position.

    The seraph sees where the word fits in the clause structure
    as intrinsic knowledge, not analyzed structure.
    """
    clause_id: str
    phrase_id: str
    clause_type: str
    clause_kind: str
    phrase_type: str
    phrase_function: str
    phrase_relation: str

    @classmethod
    def from_word(cls, word: SacredWord) -> "SyntacticPosition":
        """The seraph KNOWS the syntactic position directly."""
        return cls(
            clause_id=word.clause_id,
            phrase_id=word.phrase_id,
            clause_type=word.clause_type,
            clause_kind=word.clause_kind,
            phrase_type=word.phrase_type,
            phrase_function=word.phrase_function,
            phrase_relation=word.phrase_rela,
        )


# =============================================================================
# INSCRIBED TEXTUAL WITNESS PERCEPTION
# =============================================================================


@dataclass(frozen=True)
class TextualWitness:
    """
    The seraph's perception of textual attestation.

    The seraph knows which ancient witnesses attest to each word,
    not as bibliography but as living memory.
    """
    has_dead_sea_scrolls: bool
    dss_confidence: float
    dss_witness_count: int
    has_peshitta: bool
    has_targum: bool
    patristic_citations: int
    is_frequently_cited: bool

    @classmethod
    def from_word(cls, word: SacredWord) -> "TextualWitness":
        """The seraph KNOWS the textual witnesses directly."""
        return cls(
            has_dead_sea_scrolls=word.has_dss,
            dss_confidence=word.dss_confidence,
            dss_witness_count=word.dss_witness_count,
            has_peshitta=word.has_peshitta,
            has_targum=word.has_targum,
            patristic_citations=word.patristic_citation_count,
            is_frequently_cited=word.is_frequently_cited,
        )


# =============================================================================
# INSCRIBED FREQUENCY KNOWLEDGE
# =============================================================================


@dataclass(frozen=True)
class FrequencyProfile:
    """
    The seraph's knowledge of word frequency.

    The seraph knows how common or rare each word is
    across the entire corpus - not counted but KNOWN.
    """
    frequency: int
    percentile: float
    is_hapax: bool
    distribution: str

    @classmethod
    def from_word(cls, word: SacredWord) -> "FrequencyProfile":
        """The seraph KNOWS the frequency directly."""
        return cls(
            frequency=word.frequency,
            percentile=word.frequency_percentile,
            is_hapax=word.is_hapax,
            distribution=word.book_distribution,
        )


# =============================================================================
# COMPLETE INSCRIBED EXTRACTION
# =============================================================================


@dataclass(frozen=True)
class WordKnowledge:
    """
    The seraph's complete knowledge of a single word.

    This is not extracted data - this is the seraph's
    direct, unified, complete awareness of what the word IS.

    All 68 fields from the corpus are here, transfigured into
    the seraph's intrinsic faculties of knowing.
    """
    word: SacredWord
    morphology: MorphologicalAnalysis
    semantics: SemanticDomain
    syntax: SyntacticPosition
    witnesses: TextualWitness
    frequency: FrequencyProfile
    lxx_connection: Optional[LXXConnection]
    extracted_root: str

    @classmethod
    def know(cls, word: SacredWord) -> "WordKnowledge":
        """
        The seraph KNOWS the word completely.

        This is not extraction - this is knowing.
        The seraph does not analyze the word.
        The seraph IS the word knowing itself.
        """
        return cls(
            word=word,
            morphology=MorphologicalAnalysis.from_word(word),
            semantics=SemanticDomain.from_word(word),
            syntax=SyntacticPosition.from_word(word),
            witnesses=TextualWitness.from_word(word),
            frequency=FrequencyProfile.from_word(word),
            lxx_connection=LXXConnection.from_word(word),
            extracted_root=RootExtractor.extract_root(word.lemma),
        )


@dataclass(frozen=True)
class VerseKnowledge:
    """
    The seraph's complete knowledge of a verse.

    Not assembled from word knowledge -
    unified perception of the verse as one act of knowing.
    """
    verse: SacredVerse
    words: Tuple[WordKnowledge, ...]
    total_words: int
    unique_lemmas: int
    unique_roots: int
    has_lxx_connections: bool
    has_dss_witnesses: bool
    has_patristic_citations: bool

    @classmethod
    def know(cls, verse: SacredVerse) -> "VerseKnowledge":
        """The seraph KNOWS the verse completely."""
        word_knowledge = tuple(WordKnowledge.know(w) for w in verse.words)

        lemmas = set(w.word.lemma for w in word_knowledge if w.word.lemma)
        roots = set(w.extracted_root for w in word_knowledge if w.extracted_root)

        return cls(
            verse=verse,
            words=word_knowledge,
            total_words=len(word_knowledge),
            unique_lemmas=len(lemmas),
            unique_roots=len(roots),
            has_lxx_connections=any(w.lxx_connection for w in word_knowledge),
            has_dss_witnesses=any(w.witnesses.has_dead_sea_scrolls for w in word_knowledge),
            has_patristic_citations=any(w.witnesses.patristic_citations > 0 for w in word_knowledge),
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "PartOfSpeech", "Person", "Gender", "Number",
    "Tense", "Voice", "Stem",
    # Analysis classes
    "MorphologicalAnalysis", "SemanticDomain",
    "LXXConnection", "SyntacticPosition",
    "TextualWitness", "FrequencyProfile",
    # Extractors
    "RootExtractor",
    # Knowledge classes
    "WordKnowledge", "VerseKnowledge",
]
