"""
BIBLOS v2 - LXX Corpus Client

Integration for accessing Septuagint texts. Provides word-level morphological
data, variant readings, and supports the master_corpus.db from MASTER_LINGUISTIC_CORPUS.

Integrates with:
- MASTER_LINGUISTIC_CORPUS/RESTRUCTURED_CORPUS/Output/master_corpus.db
- Rahlfs-Hanhart LXX morphologically tagged text
- Swete's Old Testament in Greek
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

from integrations.base import (
    BaseCorpusIntegration,
    Language,
    TextType,
    VerseData,
    WordData,
    MorphologyData
)

logger = logging.getLogger(__name__)


@dataclass
class LXXWord:
    """A word from the LXX text."""
    position: int
    text: str                      # Greek text (Unicode)
    lemma: str                     # Dictionary form
    morphology: Dict[str, str]     # POS, case, number, gender, tense, etc.
    gloss: str                     # English translation
    translit: str                  # Transliteration
    semantic_domains: List[str]
    strongs: Optional[str]         # Strong's number if available


@dataclass
class LXXVerse:
    """A verse from the LXX."""
    verse_id: str                  # Standard form: BOOK.CHAPTER.VERSE
    lxx_verse_id: str              # LXX-specific numbering
    text_full: str                 # Full verse text
    words: List[LXXWord]
    variant_readings: List[Dict]   # Manuscript variants
    critical_apparatus: str        # Notes from critical edition


class LXXCorpusClient(BaseCorpusIntegration):
    """
    Client for accessing Septuagint corpus data.

    Primary source: MASTER_LINGUISTIC_CORPUS SQLite database
    Fallback: JSON corpus files
    """

    # Book name mappings (LXX sometimes uses different names)
    LXX_BOOK_NAMES = {
        "GEN": "Genesis",
        "EXO": "Exodus",
        "LEV": "Leviticus",
        "NUM": "Numbers",
        "DEU": "Deuteronomy",
        "JOS": "Joshua",
        "JDG": "Judges",
        "RUT": "Ruth",
        "1SA": "1Kingdoms",      # LXX naming
        "2SA": "2Kingdoms",
        "1KI": "3Kingdoms",
        "2KI": "4Kingdoms",
        "1CH": "1Chronicles",
        "2CH": "2Chronicles",
        "EZR": "Ezra",
        "NEH": "Nehemiah",
        "EST": "Esther",
        "JOB": "Job",
        "PSA": "Psalms",
        "PRO": "Proverbs",
        "ECC": "Ecclesiastes",
        "SNG": "SongOfSongs",
        "ISA": "Isaiah",
        "JER": "Jeremiah",
        "LAM": "Lamentations",
        "EZK": "Ezekiel",
        "DAN": "Daniel",
        "HOS": "Hosea",
        "JOL": "Joel",
        "AMO": "Amos",
        "OBA": "Obadiah",
        "JON": "Jonah",
        "MIC": "Micah",
        "NAH": "Nahum",
        "HAB": "Habakkuk",
        "ZEP": "Zephaniah",
        "HAG": "Haggai",
        "ZEC": "Zechariah",
        "MAL": "Malachi",
    }

    def __init__(
        self,
        corpus_db_path: Optional[str] = None,
        json_fallback_path: Optional[str] = None
    ):
        super().__init__(corpus_db_path)
        self.db_path = Path(corpus_db_path) if corpus_db_path else None
        self.fallback_path = Path(json_fallback_path) if json_fallback_path else None
        self._connection: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize database connection."""
        if not HAS_AIOSQLITE:
            self.logger.warning("aiosqlite not available, LXX corpus queries will be limited")
            self._initialized = True
            return

        if self.db_path and self.db_path.exists():
            self._connection = await aiosqlite.connect(str(self.db_path))
            self._connection.row_factory = aiosqlite.Row
            self.logger.info(f"Connected to LXX corpus: {self.db_path}")
        else:
            self.logger.warning(f"LXX corpus database not found: {self.db_path}")

        self._initialized = True

    async def cleanup(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
        await super().cleanup()

    async def get_verse(self, verse_id: str) -> Optional[VerseData]:
        """
        Get verse data including words and morphology.

        Args:
            verse_id: Standard verse ID (e.g., "GEN.1.1")

        Returns:
            VerseData with verse text, words, and metadata
        """
        if not self._initialized:
            await self.initialize()

        parsed = self._parse_verse_id(verse_id)
        book = parsed["book"]
        chapter = parsed["chapter"]
        verse = parsed["verse"]

        if self._connection:
            return await self._get_verse_from_db(book, chapter, verse, verse_id)
        else:
            return await self._get_verse_fallback(book, chapter, verse, verse_id)

    async def _get_verse_from_db(
        self,
        book: str,
        chapter: int,
        verse: int,
        verse_id: str
    ) -> Optional[VerseData]:
        """Get verse from SQLite database."""
        query = """
            SELECT
                v.id, v.book, v.chapter, v.verse, v.text,
                v.lxx_book, v.lxx_chapter, v.lxx_verse
            FROM lxx_verses v
            WHERE v.book = ? AND v.chapter = ? AND v.verse = ?
        """
        async with self._connection.execute(query, (book, chapter, verse)) as cursor:
            row = await cursor.fetchone()

        if not row:
            self.logger.warning(f"Verse not found: {verse_id}")
            return None

        # Query words for this verse
        word_query = """
            SELECT
                w.position, w.text, w.lemma, w.morph_code,
                w.gloss, w.translit, w.strongs,
                w.semantic_domain
            FROM lxx_words w
            WHERE w.verse_id = ?
            ORDER BY w.position
        """
        async with self._connection.execute(word_query, (row["id"],)) as cursor:
            word_rows = await cursor.fetchall()

        words = []
        for w in word_rows:
            morph = self._parse_morph_code(w["morph_code"])
            words.append(WordData(
                word_id=f"{verse_id}:{w['position']}",
                surface_form=w["text"],
                lemma=w["lemma"],
                language=Language.GREEK,
                morphology=MorphologyData(
                    part_of_speech=morph.get("pos", "unknown"),
                    person=morph.get("person"),
                    number=morph.get("number"),
                    gender=morph.get("gender"),
                    case=morph.get("case"),
                    tense=morph.get("tense"),
                    voice=morph.get("voice"),
                    mood=morph.get("mood"),
                    raw_code=w["morph_code"]
                ),
                position=w["position"],
                transliteration=w["translit"],
                gloss=w["gloss"],
                strongs=w["strongs"]
            ))

        return VerseData(
            verse_id=verse_id,
            book=book,
            chapter=chapter,
            verse=verse,
            text=row["text"],
            language=Language.GREEK,
            text_type=TextType.SEPTUAGINT,
            words=words,
            metadata={
                "lxx_book": row["lxx_book"] if row["lxx_book"] else book,
                "lxx_chapter": row["lxx_chapter"] if row["lxx_chapter"] else chapter,
                "lxx_verse": row["lxx_verse"] if row["lxx_verse"] else verse,
            }
        )

    async def _get_verse_fallback(
        self,
        book: str,
        chapter: int,
        verse: int,
        verse_id: str
    ) -> Optional[VerseData]:
        """Fallback when database is not available."""
        return VerseData(
            verse_id=verse_id,
            book=book,
            chapter=chapter,
            verse=verse,
            text="",
            language=Language.GREEK,
            text_type=TextType.SEPTUAGINT,
            words=[],
            metadata={}
        )

    async def get_verse_text(self, verse_id: str) -> Optional[str]:
        """Get just the verse text without word-level data."""
        verse_data = await self.get_verse(verse_id)
        if verse_data:
            return verse_data.text
        return None

    async def get_verse_as_dict(self, verse_id: str) -> Dict[str, Any]:
        """Get verse data as dictionary for compatibility with extractor."""
        verse_data = await self.get_verse(verse_id)
        if not verse_data:
            return {"verse_id": verse_id, "text": "", "words": []}

        words = []
        for w in verse_data.words:
            words.append({
                "text": w.surface_form,
                "lemma": w.lemma,
                "translit": w.transliteration or "",
                "gloss": w.gloss or "",
                "morphology": w.morphology.to_dict() if w.morphology else {},
                "semantic_domains": [],
                "strongs": w.strongs
            })

        return {
            "verse_id": verse_id,
            "text": verse_data.text,
            "words": words
        }

    async def get_verses(
        self,
        book: str,
        chapter: Optional[int] = None
    ) -> List[VerseData]:
        """Get verses for a book/chapter."""
        if not self._initialized:
            await self.initialize()

        if not self._connection:
            return []

        book = self._normalize_book_code(book)

        if chapter:
            query = """
                SELECT v.book, v.chapter, v.verse, v.text
                FROM lxx_verses v
                WHERE v.book = ? AND v.chapter = ?
                ORDER BY v.verse
            """
            params = (book, chapter)
        else:
            query = """
                SELECT v.book, v.chapter, v.verse, v.text
                FROM lxx_verses v
                WHERE v.book = ?
                ORDER BY v.chapter, v.verse
            """
            params = (book,)

        async with self._connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        return [
            VerseData(
                verse_id=f"{r['book']}.{r['chapter']}.{r['verse']}",
                book=r["book"],
                chapter=r["chapter"],
                verse=r["verse"],
                text=r["text"],
                language=Language.GREEK,
                text_type=TextType.SEPTUAGINT,
                words=[],
                metadata={}
            )
            for r in rows
        ]

    async def search(
        self,
        query: str,
        book: Optional[str] = None,
        limit: int = 100,
        **kwargs
    ) -> List[VerseData]:
        """Search for Greek text patterns."""
        if not self._initialized:
            await self.initialize()

        if not self._connection:
            return []

        if book:
            book = self._normalize_book_code(book)
            sql = """
                SELECT v.book, v.chapter, v.verse, v.text
                FROM lxx_verses v
                WHERE v.text LIKE ? AND v.book = ?
                LIMIT ?
            """
            params = (f"%{query}%", book, limit)
        else:
            sql = """
                SELECT v.book, v.chapter, v.verse, v.text
                FROM lxx_verses v
                WHERE v.text LIKE ?
                LIMIT ?
            """
            params = (f"%{query}%", limit)

        async with self._connection.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        return [
            VerseData(
                verse_id=f"{r['book']}.{r['chapter']}.{r['verse']}",
                book=r["book"],
                chapter=r["chapter"],
                verse=r["verse"],
                text=r["text"],
                language=Language.GREEK,
                text_type=TextType.SEPTUAGINT,
                words=[],
                metadata={}
            )
            for r in rows
        ]

    async def get_word_data(
        self,
        verse_id: str,
        word_position: int
    ) -> Optional[WordData]:
        """Get detailed word data."""
        verse_data = await self.get_verse(verse_id)
        if not verse_data:
            return None

        for word in verse_data.words:
            if word.position == word_position:
                return word

        return None

    async def get_word_occurrences(
        self,
        lemma: str,
        limit: int = 500
    ) -> List[Dict]:
        """Find all occurrences of a Greek lemma."""
        if not self._initialized:
            await self.initialize()

        if not self._connection:
            return []

        query = """
            SELECT
                v.book, v.chapter, v.verse, w.text, w.morph_code
            FROM lxx_words w
            JOIN lxx_verses v ON w.verse_id = v.id
            WHERE w.lemma = ?
            LIMIT ?
        """
        async with self._connection.execute(query, (lemma, limit)) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "verse_id": f"{r['book']}.{r['chapter']}.{r['verse']}",
                "form": r["text"],
                "morphology": self._parse_morph_code(r["morph_code"])
            }
            for r in rows
        ]

    async def get_variants(self, verse_id: str) -> List[Dict]:
        """Get manuscript variants for a verse."""
        if not self._initialized:
            await self.initialize()

        if not self._connection:
            return []

        parsed = self._parse_verse_id(verse_id)
        book, chapter, verse = parsed["book"], parsed["chapter"], parsed["verse"]

        query = """
            SELECT
                var.manuscript, var.reading, var.notes,
                m.century, m.type
            FROM lxx_variants var
            JOIN lxx_verses v ON var.verse_id = v.id
            LEFT JOIN manuscripts m ON var.manuscript = m.id
            WHERE v.book = ? AND v.chapter = ? AND v.verse = ?
        """
        try:
            async with self._connection.execute(query, (book, chapter, verse)) as cursor:
                rows = await cursor.fetchall()

            return [
                {
                    "manuscript": r["manuscript"],
                    "reading": r["reading"],
                    "notes": r["notes"],
                    "century": r["century"],
                    "type": r["type"]
                }
                for r in rows
            ]
        except Exception:
            return []

    def get_supported_books(self) -> List[str]:
        """Get list of supported book codes."""
        return list(self.LXX_BOOK_NAMES.keys())

    def get_language(self) -> Language:
        """Get the primary language of this corpus."""
        return Language.GREEK

    def _parse_morph_code(self, code: str) -> Dict[str, str]:
        """Parse morphological code into structured dict."""
        if not code:
            return {}

        morph = {}

        # Common positional encoding (e.g., Robinson's codes)
        pos_map = {
            'N': 'noun', 'V': 'verb', 'A': 'adjective', 'D': 'adverb',
            'P': 'preposition', 'C': 'conjunction', 'R': 'pronoun',
            'T': 'article', 'X': 'particle', 'I': 'interjection'
        }

        case_map = {'N': 'nominative', 'G': 'genitive', 'D': 'dative', 'A': 'accusative', 'V': 'vocative'}
        number_map = {'S': 'singular', 'P': 'plural', 'D': 'dual'}
        gender_map = {'M': 'masculine', 'F': 'feminine', 'N': 'neuter'}
        tense_map = {'P': 'present', 'I': 'imperfect', 'F': 'future', 'A': 'aorist', 'X': 'perfect', 'Y': 'pluperfect'}
        voice_map = {'A': 'active', 'M': 'middle', 'P': 'passive'}
        mood_map = {'I': 'indicative', 'S': 'subjunctive', 'O': 'optative', 'M': 'imperative', 'N': 'infinitive', 'P': 'participle'}

        if len(code) >= 1 and code[0] in pos_map:
            morph['pos'] = pos_map[code[0]]

        # Parse based on POS
        if morph.get('pos') == 'verb' and len(code) >= 6:
            if len(code) > 1 and code[1] in tense_map:
                morph['tense'] = tense_map[code[1]]
            if len(code) > 2 and code[2] in voice_map:
                morph['voice'] = voice_map[code[2]]
            if len(code) > 3 and code[3] in mood_map:
                morph['mood'] = mood_map[code[3]]
            if len(code) > 4 and code[4] in number_map:
                morph['number'] = number_map[code[4]]
            if len(code) > 5 and code[5] in {'1', '2', '3'}:
                morph['person'] = code[5]

        elif morph.get('pos') in ('noun', 'adjective', 'article', 'pronoun') and len(code) >= 4:
            if len(code) > 1 and code[1] in case_map:
                morph['case'] = case_map[code[1]]
            if len(code) > 2 and code[2] in number_map:
                morph['number'] = number_map[code[2]]
            if len(code) > 3 and code[3] in gender_map:
                morph['gender'] = gender_map[code[3]]

        return morph
