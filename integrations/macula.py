"""
BIBLOS v2 - Macula Integration

Integration with Macula Greek and Hebrew morphological data.
"""
import asyncio
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
from functools import lru_cache

from integrations.base import (
    BaseCorpusIntegration,
    VerseData,
    WordData,
    MorphologyData,
    Language,
    TextType
)


class MaculaIntegration(BaseCorpusIntegration):
    """
    Macula corpus integration.

    Provides access to:
    - macula-greek: Greek NT with detailed morphology
    - macula-hebrew: Hebrew OT with syntax trees
    """

    # Greek morphology code mappings
    GREEK_POS_MAP = {
        "N": "noun",
        "V": "verb",
        "A": "adjective",
        "D": "adverb",
        "P": "preposition",
        "C": "conjunction",
        "RA": "article",
        "RD": "demonstrative",
        "RI": "interrogative",
        "RR": "relative",
        "RP": "personal",
        "RX": "indefinite",
        "X": "particle",
        "I": "interjection"
    }

    GREEK_CASE_MAP = {
        "N": "nominative",
        "G": "genitive",
        "D": "dative",
        "A": "accusative",
        "V": "vocative"
    }

    GREEK_NUMBER_MAP = {
        "S": "singular",
        "P": "plural"
    }

    GREEK_GENDER_MAP = {
        "M": "masculine",
        "F": "feminine",
        "N": "neuter"
    }

    GREEK_TENSE_MAP = {
        "P": "present",
        "I": "imperfect",
        "F": "future",
        "A": "aorist",
        "X": "perfect",
        "Y": "pluperfect"
    }

    GREEK_VOICE_MAP = {
        "A": "active",
        "M": "middle",
        "P": "passive"
    }

    GREEK_MOOD_MAP = {
        "I": "indicative",
        "S": "subjunctive",
        "O": "optative",
        "M": "imperative",
        "N": "infinitive",
        "P": "participle"
    }

    # Hebrew morphology mappings
    HEBREW_POS_MAP = {
        "verb": "verb",
        "noun": "noun",
        "adjective": "adjective",
        "adverb": "adverb",
        "preposition": "preposition",
        "conjunction": "conjunction",
        "particle": "particle",
        "interjection": "interjection",
        "pronoun": "pronoun",
        "suffix": "suffix"
    }

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        corpus_type: str = "greek"
    ):
        super().__init__(corpus_path)
        self.corpus_type = corpus_type
        self._data: Dict[str, Any] = {}
        self._index: Dict[str, str] = {}

    async def initialize(self) -> None:
        """Initialize Macula data access."""
        self.logger.info(f"Initializing Macula integration ({self.corpus_type})")

        if self.corpus_path:
            await self._load_corpus(Path(self.corpus_path))
        else:
            # Try default paths
            default_paths = [
                Path("data/macula-greek"),
                Path("data/macula-hebrew"),
                Path("../MASTER_LINGUISTIC_CORPUS/macula-greek"),
                Path("../MASTER_LINGUISTIC_CORPUS/macula-hebrew")
            ]

            for path in default_paths:
                if path.exists():
                    await self._load_corpus(path)
                    break

        self._initialized = True
        self.logger.info(f"Macula initialized with {len(self._data)} entries")

    async def _load_corpus(self, corpus_path: Path) -> None:
        """Load Macula corpus data."""
        self.logger.info(f"Loading Macula corpus from {corpus_path}")

        try:
            # Look for different file formats
            json_files = list(corpus_path.glob("**/*.json"))
            xml_files = list(corpus_path.glob("**/*.xml"))

            if json_files:
                await self._load_json_files(json_files)
            elif xml_files:
                await self._load_xml_files(xml_files)

        except Exception as e:
            self.logger.error(f"Failed to load corpus: {e}")

    async def _load_json_files(self, files: List[Path]) -> None:
        """Load JSON format Macula files."""
        for file_path in files[:100]:  # Limit initial load
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Index by verse ID
                if isinstance(data, list):
                    for item in data:
                        verse_id = item.get("verse_id") or item.get("ref")
                        if verse_id:
                            self._data[verse_id] = item
                            self._index[verse_id] = str(file_path)
                elif isinstance(data, dict):
                    verse_id = data.get("verse_id") or data.get("ref")
                    if verse_id:
                        self._data[verse_id] = data
                        self._index[verse_id] = str(file_path)

            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")

    async def _load_xml_files(self, files: List[Path]) -> None:
        """Load XML format Macula files."""
        for file_path in files[:100]:  # Limit initial load
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                # Parse based on structure
                for verse in root.findall(".//verse"):
                    verse_id = verse.get("ref") or verse.get("osisID")
                    if verse_id:
                        self._data[verse_id] = self._parse_xml_verse(verse)
                        self._index[verse_id] = str(file_path)

            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")

    def _parse_xml_verse(self, verse_elem: ET.Element) -> Dict[str, Any]:
        """Parse XML verse element to dict."""
        words = []

        for i, word in enumerate(verse_elem.findall(".//w")):
            word_data = {
                "surface": word.text or "",
                "lemma": word.get("lemma", ""),
                "morph": word.get("morph", ""),
                "strongs": word.get("strong", word.get("strongs", "")),
                "gloss": word.get("gloss", ""),
                "position": i
            }
            words.append(word_data)

        return {
            "verse_id": verse_elem.get("ref") or verse_elem.get("osisID"),
            "text": verse_elem.get("text") or " ".join(w.get("surface", "") for w in words),
            "words": words
        }

    async def get_verse(self, verse_id: str) -> Optional[VerseData]:
        """Get verse data by ID."""
        if not self._initialized:
            await self.initialize()

        # Normalize verse ID
        verse_id = self._normalize_verse_id(verse_id)

        # Check cache
        if verse_id in self._data:
            return self._convert_to_verse_data(verse_id, self._data[verse_id])

        # Try to load from file
        await self._lazy_load_verse(verse_id)

        if verse_id in self._data:
            return self._convert_to_verse_data(verse_id, self._data[verse_id])

        return None

    def _normalize_verse_id(self, verse_id: str) -> str:
        """Normalize verse ID format."""
        # Convert various formats to standard
        verse_id = verse_id.strip()

        # Handle "Gen 1:1" format
        if " " in verse_id and ":" in verse_id:
            parts = verse_id.split()
            book = self._normalize_book_code(parts[0])
            cv = parts[1].split(":")
            return f"{book}.{cv[0]}.{cv[1]}"

        # Handle "GEN.1.1" format
        if "." in verse_id:
            parts = verse_id.split(".")
            if len(parts) >= 3:
                book = self._normalize_book_code(parts[0])
                return f"{book}.{parts[1]}.{parts[2]}"

        return verse_id

    async def _lazy_load_verse(self, verse_id: str) -> None:
        """Lazy load verse data from corpus files."""
        parsed = self._parse_verse_id(verse_id)
        book = parsed["book"]

        if not self.corpus_path:
            return

        corpus_path = Path(self.corpus_path)

        # Try common file naming patterns
        possible_files = [
            corpus_path / f"{book}.json",
            corpus_path / f"{book.lower()}.json",
            corpus_path / "books" / f"{book}.json",
            corpus_path / f"{book}.xml",
            corpus_path / "lowfat" / f"{book.lower()}.xml"
        ]

        for file_path in possible_files:
            if file_path.exists():
                if file_path.suffix == ".json":
                    await self._load_json_files([file_path])
                else:
                    await self._load_xml_files([file_path])
                break

    def _convert_to_verse_data(
        self,
        verse_id: str,
        raw_data: Dict[str, Any]
    ) -> VerseData:
        """Convert raw Macula data to VerseData."""
        parsed = self._parse_verse_id(verse_id)
        book = parsed["book"]

        # Determine language and text type
        is_greek = self.corpus_type == "greek" or book in [
            "MAT", "MRK", "LUK", "JHN", "ACT",
            "ROM", "1CO", "2CO", "GAL", "EPH",
            "PHP", "COL", "1TH", "2TH", "1TI",
            "2TI", "TIT", "PHM", "HEB", "JAS",
            "1PE", "2PE", "1JN", "2JN", "3JN",
            "JUD", "REV"
        ]

        language = Language.GREEK if is_greek else Language.HEBREW
        text_type = TextType.NA28 if is_greek else TextType.MASORETIC

        # Convert words
        words = []
        raw_words = raw_data.get("words", [])

        for i, raw_word in enumerate(raw_words):
            word_data = self._convert_word(raw_word, i, language)
            words.append(word_data)

        return VerseData(
            verse_id=verse_id,
            book=book,
            chapter=parsed["chapter"],
            verse=parsed["verse"],
            text=raw_data.get("text", " ".join(w.surface_form for w in words)),
            language=language,
            text_type=text_type,
            words=words,
            metadata={
                "source": "macula",
                "corpus_type": self.corpus_type,
                "file": self._index.get(verse_id)
            }
        )

    def _convert_word(
        self,
        raw_word: Dict[str, Any],
        position: int,
        language: Language
    ) -> WordData:
        """Convert raw word data to WordData."""
        # Parse morphology code
        morph_code = raw_word.get("morph", "")
        morphology = self._parse_morphology(morph_code, language)

        return WordData(
            word_id=raw_word.get("id", str(position)),
            surface_form=raw_word.get("surface", raw_word.get("text", "")),
            lemma=raw_word.get("lemma", ""),
            language=language,
            morphology=morphology,
            position=position,
            transliteration=raw_word.get("translit"),
            gloss=raw_word.get("gloss"),
            strongs=raw_word.get("strongs", raw_word.get("strong")),
            syntax_role=raw_word.get("role", raw_word.get("function")),
            clause_id=raw_word.get("clause_id"),
            phrase_id=raw_word.get("phrase_id")
        )

    def _parse_morphology(
        self,
        code: str,
        language: Language
    ) -> MorphologyData:
        """Parse morphology code."""
        if language == Language.GREEK:
            return self._parse_greek_morphology(code)
        else:
            return self._parse_hebrew_morphology(code)

    def _parse_greek_morphology(self, code: str) -> MorphologyData:
        """Parse Greek morphology code (Robinson format)."""
        if not code:
            return MorphologyData(part_of_speech="unknown")

        # Robinson code format: V-PAI-3S, N-NSM, etc.
        parts = code.replace("-", "").upper()

        pos = self.GREEK_POS_MAP.get(parts[0:1], "unknown")

        # Parse based on POS
        person = None
        number = None
        gender = None
        case = None
        tense = None
        voice = None
        mood = None

        if pos == "verb" and len(parts) >= 4:
            tense = self.GREEK_TENSE_MAP.get(parts[1:2])
            voice = self.GREEK_VOICE_MAP.get(parts[2:3])
            mood = self.GREEK_MOOD_MAP.get(parts[3:4])
            if len(parts) >= 5:
                person = parts[4:5]
            if len(parts) >= 6:
                number = self.GREEK_NUMBER_MAP.get(parts[5:6])

        elif pos in ["noun", "adjective", "article", "pronoun"]:
            if len(parts) >= 2:
                case = self.GREEK_CASE_MAP.get(parts[1:2])
            if len(parts) >= 3:
                number = self.GREEK_NUMBER_MAP.get(parts[2:3])
            if len(parts) >= 4:
                gender = self.GREEK_GENDER_MAP.get(parts[3:4])

        return MorphologyData(
            part_of_speech=pos,
            person=person,
            number=number,
            gender=gender,
            case=case,
            tense=tense,
            voice=voice,
            mood=mood,
            raw_code=code
        )

    def _parse_hebrew_morphology(self, code: str) -> MorphologyData:
        """Parse Hebrew morphology code."""
        if not code:
            return MorphologyData(part_of_speech="unknown")

        # Hebrew codes vary by source
        # Common format: HVqp3ms (Hebrew Verb qal perfect 3rd masculine singular)
        code = code.upper()

        pos = "unknown"
        stem = None
        tense = None
        person = None
        gender = None
        number = None
        state = None

        # Parse based on prefix
        if code.startswith("HV") or code.startswith("V"):
            pos = "verb"
            # Extract stem
            stem_pos = 2 if code.startswith("HV") else 1
            if len(code) > stem_pos:
                stem_codes = {
                    "Q": "qal",
                    "N": "niphal",
                    "P": "piel",
                    "U": "pual",
                    "H": "hiphil",
                    "O": "hophal",
                    "T": "hithpael"
                }
                stem = stem_codes.get(code[stem_pos:stem_pos+1])

        elif code.startswith("HN") or code.startswith("N"):
            pos = "noun"
        elif code.startswith("HA") or code.startswith("A"):
            pos = "adjective"
        elif code.startswith("HP") or code.startswith("P"):
            pos = "preposition"
        elif code.startswith("HC") or code.startswith("C"):
            pos = "conjunction"

        return MorphologyData(
            part_of_speech=pos,
            person=person,
            number=number,
            gender=gender,
            state=state,
            stem=stem,
            tense=tense,
            raw_code=code
        )

    async def get_verses(
        self,
        book: str,
        chapter: Optional[int] = None
    ) -> List[VerseData]:
        """Get all verses for a book/chapter."""
        if not self._initialized:
            await self.initialize()

        book = self._normalize_book_code(book)
        results = []

        # Filter cached data
        for verse_id, data in self._data.items():
            parsed = self._parse_verse_id(verse_id)
            if parsed["book"] == book:
                if chapter is None or parsed["chapter"] == chapter:
                    verse = self._convert_to_verse_data(verse_id, data)
                    results.append(verse)

        # Sort by chapter and verse
        results.sort(key=lambda v: (v.chapter, v.verse))
        return results

    async def search(
        self,
        query: str,
        **kwargs
    ) -> List[VerseData]:
        """Search the corpus."""
        if not self._initialized:
            await self.initialize()

        results = []
        query_lower = query.lower()
        limit = kwargs.get("limit", 100)

        for verse_id, data in self._data.items():
            # Search in text
            text = data.get("text", "")
            if query_lower in text.lower():
                verse = self._convert_to_verse_data(verse_id, data)
                results.append(verse)
                if len(results) >= limit:
                    break

            # Search in lemmas
            for word in data.get("words", []):
                lemma = word.get("lemma", "")
                if query_lower in lemma.lower():
                    verse = self._convert_to_verse_data(verse_id, data)
                    if verse not in results:
                        results.append(verse)
                    break

            if len(results) >= limit:
                break

        return results

    async def get_word_data(
        self,
        verse_id: str,
        word_position: int
    ) -> Optional[WordData]:
        """Get detailed word data."""
        verse = await self.get_verse(verse_id)
        if verse and word_position < len(verse.words):
            return verse.words[word_position]
        return None

    def get_supported_books(self) -> List[str]:
        """Get list of supported book codes."""
        if self.corpus_type == "greek":
            return [
                "MAT", "MRK", "LUK", "JHN", "ACT",
                "ROM", "1CO", "2CO", "GAL", "EPH",
                "PHP", "COL", "1TH", "2TH", "1TI",
                "2TI", "TIT", "PHM", "HEB", "JAS",
                "1PE", "2PE", "1JN", "2JN", "3JN",
                "JUD", "REV"
            ]
        else:
            return [
                "GEN", "EXO", "LEV", "NUM", "DEU",
                "JOS", "JDG", "RUT", "1SA", "2SA",
                "1KI", "2KI", "1CH", "2CH", "EZR",
                "NEH", "EST", "JOB", "PSA", "PRO",
                "ECC", "SNG", "ISA", "JER", "LAM",
                "EZK", "DAN", "HOS", "JOL", "AMO",
                "OBA", "JON", "MIC", "NAH", "HAB",
                "ZEP", "HAG", "ZEC", "MAL"
            ]

    def get_language(self) -> Language:
        """Get the primary language of this corpus."""
        return Language.GREEK if self.corpus_type == "greek" else Language.HEBREW

    async def get_syntax_tree(
        self,
        verse_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get syntactic tree for a verse (lowfat format)."""
        if not self._initialized:
            await self.initialize()

        verse_id = self._normalize_verse_id(verse_id)

        if verse_id not in self._data:
            return None

        data = self._data[verse_id]

        # Build tree from word data
        tree = {
            "verse_id": verse_id,
            "words": []
        }

        for word in data.get("words", []):
            word_node = {
                "surface": word.get("surface", ""),
                "lemma": word.get("lemma", ""),
                "morph": word.get("morph", ""),
                "role": word.get("role"),
                "clause": word.get("clause_id"),
                "phrase": word.get("phrase_id")
            }
            tree["words"].append(word_node)

        return tree

    async def cleanup(self) -> None:
        """Cleanup Macula resources."""
        self._data.clear()
        self._index.clear()
        self._initialized = False
        self.logger.info("Macula integration cleaned up")
