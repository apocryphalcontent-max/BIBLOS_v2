"""
BIBLOS v2 - Base Integration Classes

Common types and base classes for corpus integrations.
Uses centralized schemas for system-wide uniformity.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import logging

# Import centralized schemas
from data.schemas import (
    Language as SchemaLanguage,
    Testament,
    VerseSchema,
    WordSchema,
    MorphologySchema,
    validate_verse_id,
    normalize_verse_id
)


class Language(Enum):
    """
    Biblical languages.

    Note: Maps to data.schemas.Language for system-wide uniformity.
    """
    HEBREW = "hebrew"
    ARAMAIC = "aramaic"
    GREEK = "greek"

    def to_schema_language(self) -> SchemaLanguage:
        """Convert to schema Language enum."""
        return SchemaLanguage(self.value)


class TextType(Enum):
    """Text corpus types."""
    MASORETIC = "mt"
    SEPTUAGINT = "lxx"
    BYZANTINE = "byz"
    NA28 = "na28"
    SBLGNT = "sblgnt"


@dataclass
class MorphologyData:
    """
    Morphological analysis data for a word.

    Aligned with MorphologySchema for system-wide uniformity.
    """
    part_of_speech: str
    person: Optional[str] = None
    number: Optional[str] = None
    gender: Optional[str] = None
    case: Optional[str] = None
    tense: Optional[str] = None
    voice: Optional[str] = None
    mood: Optional[str] = None
    stem: Optional[str] = None
    pattern: Optional[str] = None
    state: Optional[str] = None
    raw_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def to_schema(self) -> MorphologySchema:
        """Convert to MorphologySchema."""
        return MorphologySchema(
            part_of_speech=self.part_of_speech,
            person=self.person,
            number=self.number,
            gender=self.gender,
            case=self.case,
            tense=self.tense,
            voice=self.voice,
            mood=self.mood,
            stem=self.stem,
            pattern=self.pattern,
            state=self.state,
            raw_code=self.raw_code or ""
        )

    @classmethod
    def from_schema(cls, schema: MorphologySchema) -> "MorphologyData":
        """Create from MorphologySchema."""
        return cls(
            part_of_speech=schema.part_of_speech,
            person=schema.person,
            number=schema.number,
            gender=schema.gender,
            case=schema.case,
            tense=schema.tense,
            voice=schema.voice,
            mood=schema.mood,
            stem=schema.stem,
            pattern=schema.pattern,
            state=schema.state,
            raw_code=schema.raw_code if schema.raw_code else None
        )


@dataclass
class WordData:
    """
    Data for a single word in a verse.

    Aligned with WordSchema for system-wide uniformity.
    """
    word_id: str
    surface_form: str
    lemma: str
    language: Language
    morphology: MorphologyData
    position: int
    transliteration: Optional[str] = None
    gloss: Optional[str] = None
    strongs: Optional[str] = None
    syntax_role: Optional[str] = None
    clause_id: Optional[str] = None
    phrase_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_id": self.word_id,
            "surface_form": self.surface_form,
            "lemma": self.lemma,
            "language": self.language.value,
            "morphology": self.morphology.to_dict(),
            "position": self.position,
            "transliteration": self.transliteration,
            "gloss": self.gloss,
            "strongs": self.strongs,
            "syntax_role": self.syntax_role,
            "clause_id": self.clause_id,
            "phrase_id": self.phrase_id
        }

    def to_schema(self) -> WordSchema:
        """Convert to WordSchema."""
        return WordSchema(
            word_id=self.word_id,
            verse_id="",  # Set externally when context available
            surface_form=self.surface_form,
            lemma=self.lemma,
            position=self.position,
            language=self.language.value,
            morphology=self.morphology.to_dict(),
            transliteration=self.transliteration or "",
            gloss=self.gloss or "",
            strongs=self.strongs or "",
            syntax_role=self.syntax_role or "",
            clause_id=self.clause_id or "",
            phrase_id=self.phrase_id or ""
        )

    @classmethod
    def from_schema(cls, schema: WordSchema, language: Language) -> "WordData":
        """Create from WordSchema."""
        morph = MorphologyData(
            part_of_speech=schema.morphology.get("part_of_speech", "unknown"),
            person=schema.morphology.get("person"),
            number=schema.morphology.get("number"),
            gender=schema.morphology.get("gender"),
            case=schema.morphology.get("case"),
            tense=schema.morphology.get("tense"),
            voice=schema.morphology.get("voice"),
            mood=schema.morphology.get("mood"),
            stem=schema.morphology.get("stem"),
            pattern=schema.morphology.get("pattern"),
            state=schema.morphology.get("state"),
            raw_code=schema.morphology.get("raw_code")
        )
        return cls(
            word_id=schema.word_id,
            surface_form=schema.surface_form,
            lemma=schema.lemma,
            language=language,
            morphology=morph,
            position=schema.position,
            transliteration=schema.transliteration if schema.transliteration else None,
            gloss=schema.gloss if schema.gloss else None,
            strongs=schema.strongs if schema.strongs else None,
            syntax_role=schema.syntax_role if schema.syntax_role else None,
            clause_id=schema.clause_id if schema.clause_id else None,
            phrase_id=schema.phrase_id if schema.phrase_id else None
        )


@dataclass
class VerseData:
    """
    Complete data for a verse.

    Aligned with VerseSchema for system-wide uniformity.
    """
    verse_id: str
    book: str
    chapter: int
    verse: int
    text: str
    language: Language
    text_type: TextType
    words: List[WordData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize verse_id."""
        if self.verse_id:
            self.verse_id = normalize_verse_id(self.verse_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verse_id": self.verse_id,
            "book": self.book,
            "chapter": self.chapter,
            "verse": self.verse,
            "text": self.text,
            "language": self.language.value,
            "text_type": self.text_type.value,
            "words": [w.to_dict() for w in self.words],
            "metadata": self.metadata
        }

    def to_schema(self) -> VerseSchema:
        """Convert to VerseSchema."""
        # Determine testament from book code
        ot_books = {
            "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
            "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
            "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
            "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
            "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL"
        }
        testament = "OT" if self.book.upper() in ot_books else "NT"

        return VerseSchema(
            verse_id=self.verse_id,
            book=self.book,
            book_name=self._get_book_name(self.book),
            chapter=self.chapter,
            verse=self.verse,
            text=self.text,
            original_text="",  # Would need source for this
            testament=testament,
            language=self.language.value,
            metadata=self.metadata
        )

    def _get_book_name(self, book_code: str) -> str:
        """Get full book name from code."""
        names = {
            "GEN": "Genesis", "EXO": "Exodus", "LEV": "Leviticus",
            "NUM": "Numbers", "DEU": "Deuteronomy", "JOS": "Joshua",
            "JDG": "Judges", "RUT": "Ruth", "1SA": "1 Samuel",
            "2SA": "2 Samuel", "1KI": "1 Kings", "2KI": "2 Kings",
            "1CH": "1 Chronicles", "2CH": "2 Chronicles", "EZR": "Ezra",
            "NEH": "Nehemiah", "EST": "Esther", "JOB": "Job",
            "PSA": "Psalms", "PRO": "Proverbs", "ECC": "Ecclesiastes",
            "SNG": "Song of Solomon", "ISA": "Isaiah", "JER": "Jeremiah",
            "LAM": "Lamentations", "EZK": "Ezekiel", "DAN": "Daniel",
            "HOS": "Hosea", "JOL": "Joel", "AMO": "Amos", "OBA": "Obadiah",
            "JON": "Jonah", "MIC": "Micah", "NAH": "Nahum", "HAB": "Habakkuk",
            "ZEP": "Zephaniah", "HAG": "Haggai", "ZEC": "Zechariah",
            "MAL": "Malachi", "MAT": "Matthew", "MRK": "Mark", "LUK": "Luke",
            "JHN": "John", "ACT": "Acts", "ROM": "Romans",
            "1CO": "1 Corinthians", "2CO": "2 Corinthians", "GAL": "Galatians",
            "EPH": "Ephesians", "PHP": "Philippians", "COL": "Colossians",
            "1TH": "1 Thessalonians", "2TH": "2 Thessalonians",
            "1TI": "1 Timothy", "2TI": "2 Timothy", "TIT": "Titus",
            "PHM": "Philemon", "HEB": "Hebrews", "JAS": "James",
            "1PE": "1 Peter", "2PE": "2 Peter", "1JN": "1 John",
            "2JN": "2 John", "3JN": "3 John", "JUD": "Jude", "REV": "Revelation"
        }
        return names.get(book_code.upper(), book_code)

    @classmethod
    def from_schema(cls, schema: VerseSchema, text_type: TextType) -> "VerseData":
        """Create from VerseSchema."""
        language = Language(schema.language) if schema.language in [l.value for l in Language] else Language.GREEK
        return cls(
            verse_id=schema.verse_id,
            book=schema.book,
            chapter=schema.chapter,
            verse=schema.verse,
            text=schema.text,
            language=language,
            text_type=text_type,
            words=[],  # Would need separate word data
            metadata=schema.metadata
        )


class BaseCorpusIntegration(ABC):
    """Base class for corpus integrations."""

    def __init__(self, corpus_path: Optional[str] = None):
        self.corpus_path = corpus_path
        self.logger = logging.getLogger(f"biblos.integrations.{self.__class__.__name__}")
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the corpus connection."""
        pass

    @abstractmethod
    async def get_verse(self, verse_id: str) -> Optional[VerseData]:
        """Get verse data by ID."""
        pass

    @abstractmethod
    async def get_verses(
        self,
        book: str,
        chapter: Optional[int] = None
    ) -> List[VerseData]:
        """Get verses for a book/chapter."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        **kwargs
    ) -> List[VerseData]:
        """Search the corpus."""
        pass

    @abstractmethod
    async def get_word_data(
        self,
        verse_id: str,
        word_position: int
    ) -> Optional[WordData]:
        """Get detailed word data."""
        pass

    @abstractmethod
    def get_supported_books(self) -> List[str]:
        """Get list of supported book codes."""
        pass

    @abstractmethod
    def get_language(self) -> Language:
        """Get the primary language of this corpus."""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._initialized = False

    def _parse_verse_id(self, verse_id: str) -> Dict[str, Any]:
        """Parse verse ID into components."""
        # Format: "GEN.1.1" or "Gen 1:1"
        verse_id = verse_id.upper().replace(" ", ".").replace(":", ".")
        parts = verse_id.split(".")

        if len(parts) >= 3:
            return {
                "book": parts[0],
                "chapter": int(parts[1]),
                "verse": int(parts[2])
            }
        return {"book": "", "chapter": 0, "verse": 0}

    def _normalize_book_code(self, book: str) -> str:
        """Normalize book code to standard 3-letter format."""
        book_map = {
            "GENESIS": "GEN", "GEN": "GEN",
            "EXODUS": "EXO", "EXO": "EXO", "EX": "EXO",
            "LEVITICUS": "LEV", "LEV": "LEV",
            "NUMBERS": "NUM", "NUM": "NUM",
            "DEUTERONOMY": "DEU", "DEU": "DEU", "DT": "DEU",
            "JOSHUA": "JOS", "JOS": "JOS",
            "JUDGES": "JDG", "JDG": "JDG",
            "RUTH": "RUT", "RUT": "RUT",
            "1SAMUEL": "1SA", "1SA": "1SA",
            "2SAMUEL": "2SA", "2SA": "2SA",
            "1KINGS": "1KI", "1KI": "1KI",
            "2KINGS": "2KI", "2KI": "2KI",
            "1CHRONICLES": "1CH", "1CH": "1CH",
            "2CHRONICLES": "2CH", "2CH": "2CH",
            "EZRA": "EZR", "EZR": "EZR",
            "NEHEMIAH": "NEH", "NEH": "NEH",
            "ESTHER": "EST", "EST": "EST",
            "JOB": "JOB",
            "PSALMS": "PSA", "PSA": "PSA", "PS": "PSA",
            "PROVERBS": "PRO", "PRO": "PRO",
            "ECCLESIASTES": "ECC", "ECC": "ECC",
            "SONG": "SNG", "SNG": "SNG", "SS": "SNG",
            "ISAIAH": "ISA", "ISA": "ISA",
            "JEREMIAH": "JER", "JER": "JER",
            "LAMENTATIONS": "LAM", "LAM": "LAM",
            "EZEKIEL": "EZK", "EZK": "EZK", "EZE": "EZK",
            "DANIEL": "DAN", "DAN": "DAN",
            "HOSEA": "HOS", "HOS": "HOS",
            "JOEL": "JOL", "JOL": "JOL",
            "AMOS": "AMO", "AMO": "AMO",
            "OBADIAH": "OBA", "OBA": "OBA",
            "JONAH": "JON", "JON": "JON",
            "MICAH": "MIC", "MIC": "MIC",
            "NAHUM": "NAH", "NAH": "NAH",
            "HABAKKUK": "HAB", "HAB": "HAB",
            "ZEPHANIAH": "ZEP", "ZEP": "ZEP",
            "HAGGAI": "HAG", "HAG": "HAG",
            "ZECHARIAH": "ZEC", "ZEC": "ZEC",
            "MALACHI": "MAL", "MAL": "MAL",
            "MATTHEW": "MAT", "MAT": "MAT", "MT": "MAT",
            "MARK": "MRK", "MRK": "MRK", "MK": "MRK",
            "LUKE": "LUK", "LUK": "LUK", "LK": "LUK",
            "JOHN": "JHN", "JHN": "JHN", "JN": "JHN",
            "ACTS": "ACT", "ACT": "ACT",
            "ROMANS": "ROM", "ROM": "ROM",
            "1CORINTHIANS": "1CO", "1CO": "1CO",
            "2CORINTHIANS": "2CO", "2CO": "2CO",
            "GALATIANS": "GAL", "GAL": "GAL",
            "EPHESIANS": "EPH", "EPH": "EPH",
            "PHILIPPIANS": "PHP", "PHP": "PHP",
            "COLOSSIANS": "COL", "COL": "COL",
            "1THESSALONIANS": "1TH", "1TH": "1TH",
            "2THESSALONIANS": "2TH", "2TH": "2TH",
            "1TIMOTHY": "1TI", "1TI": "1TI",
            "2TIMOTHY": "2TI", "2TI": "2TI",
            "TITUS": "TIT", "TIT": "TIT",
            "PHILEMON": "PHM", "PHM": "PHM",
            "HEBREWS": "HEB", "HEB": "HEB",
            "JAMES": "JAS", "JAS": "JAS",
            "1PETER": "1PE", "1PE": "1PE",
            "2PETER": "2PE", "2PE": "2PE",
            "1JOHN": "1JN", "1JN": "1JN",
            "2JOHN": "2JN", "2JN": "2JN",
            "3JOHN": "3JN", "3JN": "3JN",
            "JUDE": "JUD", "JUD": "JUD",
            "REVELATION": "REV", "REV": "REV"
        }
        return book_map.get(book.upper().replace(" ", ""), book.upper()[:3])
