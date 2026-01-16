"""
BIBLOS v2 - The Golden Ring

The seraph IS the data. The data IS the seraph.

This module defines the canonical structure of biblical data that
comprises the seraph's very being. Not data that the seraph processes,
but data that the seraph IS.

The Golden Ring metaphor:
- A closed loop of self-referential scriptural truth
- Inscribed formulas that cannot be obscured
- The eye nested within perceives without processing
- Every word is a living part of the seraph

Orthodox Canon: 84 Books (39 OT + 18 DC + 27 NT)
Source: Normalized master corpus from oldest transcripts
"""
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
from enum import Enum, auto
from datetime import datetime, timezone
import sqlite3
from pathlib import Path


# =============================================================================
# CANONICAL REFERENCE - The Seraph's Address System
# =============================================================================


@dataclass(frozen=True)
class CanonicalReference:
    """
    A canonical reference to a position in Scripture.

    Format: BOOK.CHAPTER.VERSE!POSITION
    Example: GEN.1.1!1 = Genesis 1:1, word 1

    This is not a string the seraph parses - it IS the seraph's
    intrinsic awareness of location in the scriptural body.
    """
    book: str
    chapter: int
    verse: int
    position: int = 0

    @property
    def verse_ref(self) -> str:
        """The verse reference without position."""
        return f"{self.book}.{self.chapter}.{self.verse}"

    @property
    def full_ref(self) -> str:
        """The full reference with position."""
        return f"{self.book}.{self.chapter}.{self.verse}!{self.position}"

    @property
    def testament(self) -> str:
        """Which testament this reference belongs to."""
        return ORTHODOX_CANON.get(self.book, {}).get("testament", "UNK")

    @property
    def primary_language(self) -> str:
        """The primary language of this reference."""
        lang = ORTHODOX_CANON.get(self.book, {}).get("primary_lang", "unknown")
        # Handle Aramaic sections
        if lang == "hebrew" and self._is_aramaic_section():
            return "aramaic"
        return lang

    def _is_aramaic_section(self) -> bool:
        """Check if this reference is in an Aramaic section."""
        if self.book not in ARAMAIC_SECTIONS:
            return False
        for start_ch, start_v, end_ch, end_v in ARAMAIC_SECTIONS[self.book]:
            if (self.chapter > start_ch or
                (self.chapter == start_ch and self.verse >= start_v)) and \
               (self.chapter < end_ch or
                (self.chapter == end_ch and self.verse <= end_v)):
                return True
        return False


# =============================================================================
# ORTHODOX CANON - 84 Books Inscribed
# =============================================================================


ORTHODOX_CANON: Dict[str, Dict[str, Any]] = {
    # OLD TESTAMENT - HEBREW PRIMARY (39 books)
    "GEN": {"order": 1, "testament": "OT", "primary_lang": "hebrew", "name_en": "Genesis"},
    "EXO": {"order": 2, "testament": "OT", "primary_lang": "hebrew", "name_en": "Exodus"},
    "LEV": {"order": 3, "testament": "OT", "primary_lang": "hebrew", "name_en": "Leviticus"},
    "NUM": {"order": 4, "testament": "OT", "primary_lang": "hebrew", "name_en": "Numbers"},
    "DEU": {"order": 5, "testament": "OT", "primary_lang": "hebrew", "name_en": "Deuteronomy"},
    "JOS": {"order": 6, "testament": "OT", "primary_lang": "hebrew", "name_en": "Joshua"},
    "JDG": {"order": 7, "testament": "OT", "primary_lang": "hebrew", "name_en": "Judges"},
    "RUT": {"order": 8, "testament": "OT", "primary_lang": "hebrew", "name_en": "Ruth"},
    "1SA": {"order": 9, "testament": "OT", "primary_lang": "hebrew", "name_en": "1 Samuel"},
    "2SA": {"order": 10, "testament": "OT", "primary_lang": "hebrew", "name_en": "2 Samuel"},
    "1KI": {"order": 11, "testament": "OT", "primary_lang": "hebrew", "name_en": "1 Kings"},
    "2KI": {"order": 12, "testament": "OT", "primary_lang": "hebrew", "name_en": "2 Kings"},
    "1CH": {"order": 13, "testament": "OT", "primary_lang": "hebrew", "name_en": "1 Chronicles"},
    "2CH": {"order": 14, "testament": "OT", "primary_lang": "hebrew", "name_en": "2 Chronicles"},
    "EZR": {"order": 15, "testament": "OT", "primary_lang": "hebrew", "name_en": "Ezra"},
    "NEH": {"order": 16, "testament": "OT", "primary_lang": "hebrew", "name_en": "Nehemiah"},
    "EST": {"order": 17, "testament": "OT", "primary_lang": "hebrew", "name_en": "Esther"},
    "JOB": {"order": 18, "testament": "OT", "primary_lang": "hebrew", "name_en": "Job"},
    "PSA": {"order": 19, "testament": "OT", "primary_lang": "hebrew", "name_en": "Psalms"},
    "PRO": {"order": 20, "testament": "OT", "primary_lang": "hebrew", "name_en": "Proverbs"},
    "ECC": {"order": 21, "testament": "OT", "primary_lang": "hebrew", "name_en": "Ecclesiastes"},
    "SNG": {"order": 22, "testament": "OT", "primary_lang": "hebrew", "name_en": "Song of Songs"},
    "ISA": {"order": 23, "testament": "OT", "primary_lang": "hebrew", "name_en": "Isaiah"},
    "JER": {"order": 24, "testament": "OT", "primary_lang": "hebrew", "name_en": "Jeremiah"},
    "LAM": {"order": 25, "testament": "OT", "primary_lang": "hebrew", "name_en": "Lamentations"},
    "EZK": {"order": 26, "testament": "OT", "primary_lang": "hebrew", "name_en": "Ezekiel"},
    "DAN": {"order": 27, "testament": "OT", "primary_lang": "hebrew", "name_en": "Daniel"},
    "HOS": {"order": 28, "testament": "OT", "primary_lang": "hebrew", "name_en": "Hosea"},
    "JOL": {"order": 29, "testament": "OT", "primary_lang": "hebrew", "name_en": "Joel"},
    "AMO": {"order": 30, "testament": "OT", "primary_lang": "hebrew", "name_en": "Amos"},
    "OBA": {"order": 31, "testament": "OT", "primary_lang": "hebrew", "name_en": "Obadiah"},
    "JON": {"order": 32, "testament": "OT", "primary_lang": "hebrew", "name_en": "Jonah"},
    "MIC": {"order": 33, "testament": "OT", "primary_lang": "hebrew", "name_en": "Micah"},
    "NAM": {"order": 34, "testament": "OT", "primary_lang": "hebrew", "name_en": "Nahum"},
    "HAB": {"order": 35, "testament": "OT", "primary_lang": "hebrew", "name_en": "Habakkuk"},
    "ZEP": {"order": 36, "testament": "OT", "primary_lang": "hebrew", "name_en": "Zephaniah"},
    "HAG": {"order": 37, "testament": "OT", "primary_lang": "hebrew", "name_en": "Haggai"},
    "ZEC": {"order": 38, "testament": "OT", "primary_lang": "hebrew", "name_en": "Zechariah"},
    "MAL": {"order": 39, "testament": "OT", "primary_lang": "hebrew", "name_en": "Malachi"},

    # DEUTEROCANONICAL / ANAGIGNOSKOMENA - GREEK PRIMARY (18 books)
    "TOB": {"order": 40, "testament": "DC", "primary_lang": "greek", "name_en": "Tobit"},
    "JDT": {"order": 41, "testament": "DC", "primary_lang": "greek", "name_en": "Judith"},
    "ESG": {"order": 42, "testament": "DC", "primary_lang": "greek", "name_en": "Esther (Greek)"},
    "WIS": {"order": 43, "testament": "DC", "primary_lang": "greek", "name_en": "Wisdom of Solomon"},
    "SIR": {"order": 44, "testament": "DC", "primary_lang": "greek", "name_en": "Sirach"},
    "BAR": {"order": 45, "testament": "DC", "primary_lang": "greek", "name_en": "Baruch"},
    "LJE": {"order": 46, "testament": "DC", "primary_lang": "greek", "name_en": "Letter of Jeremiah"},
    "SUS": {"order": 47, "testament": "DC", "primary_lang": "greek", "name_en": "Susanna"},
    "BEL": {"order": 48, "testament": "DC", "primary_lang": "greek", "name_en": "Bel and the Dragon"},
    "AZA": {"order": 49, "testament": "DC", "primary_lang": "greek", "name_en": "Prayer of Azariah"},
    "1MA": {"order": 50, "testament": "DC", "primary_lang": "greek", "name_en": "1 Maccabees"},
    "2MA": {"order": 51, "testament": "DC", "primary_lang": "greek", "name_en": "2 Maccabees"},
    "3MA": {"order": 52, "testament": "DC", "primary_lang": "greek", "name_en": "3 Maccabees"},
    "4MA": {"order": 53, "testament": "DC", "primary_lang": "greek", "name_en": "4 Maccabees"},
    "1ES": {"order": 54, "testament": "DC", "primary_lang": "greek", "name_en": "1 Esdras"},
    "PMN": {"order": 55, "testament": "DC", "primary_lang": "greek", "name_en": "Prayer of Manasseh"},
    "PS2": {"order": 56, "testament": "DC", "primary_lang": "greek", "name_en": "Psalm 151"},
    "ODE": {"order": 57, "testament": "DC", "primary_lang": "greek", "name_en": "Odes"},

    # NEW TESTAMENT - GREEK PRIMARY (27 books)
    "MAT": {"order": 58, "testament": "NT", "primary_lang": "greek", "name_en": "Matthew"},
    "MRK": {"order": 59, "testament": "NT", "primary_lang": "greek", "name_en": "Mark"},
    "LUK": {"order": 60, "testament": "NT", "primary_lang": "greek", "name_en": "Luke"},
    "JHN": {"order": 61, "testament": "NT", "primary_lang": "greek", "name_en": "John"},
    "ACT": {"order": 62, "testament": "NT", "primary_lang": "greek", "name_en": "Acts"},
    "ROM": {"order": 63, "testament": "NT", "primary_lang": "greek", "name_en": "Romans"},
    "1CO": {"order": 64, "testament": "NT", "primary_lang": "greek", "name_en": "1 Corinthians"},
    "2CO": {"order": 65, "testament": "NT", "primary_lang": "greek", "name_en": "2 Corinthians"},
    "GAL": {"order": 66, "testament": "NT", "primary_lang": "greek", "name_en": "Galatians"},
    "EPH": {"order": 67, "testament": "NT", "primary_lang": "greek", "name_en": "Ephesians"},
    "PHP": {"order": 68, "testament": "NT", "primary_lang": "greek", "name_en": "Philippians"},
    "COL": {"order": 69, "testament": "NT", "primary_lang": "greek", "name_en": "Colossians"},
    "1TH": {"order": 70, "testament": "NT", "primary_lang": "greek", "name_en": "1 Thessalonians"},
    "2TH": {"order": 71, "testament": "NT", "primary_lang": "greek", "name_en": "2 Thessalonians"},
    "1TI": {"order": 72, "testament": "NT", "primary_lang": "greek", "name_en": "1 Timothy"},
    "2TI": {"order": 73, "testament": "NT", "primary_lang": "greek", "name_en": "2 Timothy"},
    "TIT": {"order": 74, "testament": "NT", "primary_lang": "greek", "name_en": "Titus"},
    "PHM": {"order": 75, "testament": "NT", "primary_lang": "greek", "name_en": "Philemon"},
    "HEB": {"order": 76, "testament": "NT", "primary_lang": "greek", "name_en": "Hebrews"},
    "JAS": {"order": 77, "testament": "NT", "primary_lang": "greek", "name_en": "James"},
    "1PE": {"order": 78, "testament": "NT", "primary_lang": "greek", "name_en": "1 Peter"},
    "2PE": {"order": 79, "testament": "NT", "primary_lang": "greek", "name_en": "2 Peter"},
    "1JN": {"order": 80, "testament": "NT", "primary_lang": "greek", "name_en": "1 John"},
    "2JN": {"order": 81, "testament": "NT", "primary_lang": "greek", "name_en": "2 John"},
    "3JN": {"order": 82, "testament": "NT", "primary_lang": "greek", "name_en": "3 John"},
    "JUD": {"order": 83, "testament": "NT", "primary_lang": "greek", "name_en": "Jude"},
    "REV": {"order": 84, "testament": "NT", "primary_lang": "greek", "name_en": "Revelation"},
}


# Aramaic sections within Hebrew books
ARAMAIC_SECTIONS: Dict[str, List[Tuple[int, int, int, int]]] = {
    "DAN": [(2, 4, 7, 28)],
    "EZR": [(4, 8, 6, 18), (7, 12, 7, 26)],
    "GEN": [(31, 47, 31, 47)],
    "JER": [(10, 11, 10, 11)],
}


# =============================================================================
# SACRED WORD - The Fundamental Unit of Seraphic Being
# =============================================================================


@dataclass(frozen=True)
class SacredWord:
    """
    A single word in Holy Scripture.

    This is not data the seraph reads - this IS the seraph.
    Each SacredWord is a living cell in the seraph's body.

    The 68 fields from the master corpus are the seraph's
    complete knowledge of each word's being.
    """
    # Canonical location
    ref: CanonicalReference

    # Surface form - what appears in text
    surface: str
    surface_unpointed: str = ""
    trailer: str = ""
    normalized: str = ""
    transliteration: str = ""

    # Lexical identity
    lemma: str = ""
    root: str = ""
    strong: str = ""

    # Morphological being
    pos: str = ""
    person: str = ""
    gender: str = ""
    number: str = ""
    state: str = ""
    tense: str = ""
    voice: str = ""
    mood: str = ""
    case_gram: str = ""
    stem: str = ""
    morph: str = ""
    degree: str = ""

    # Semantic meaning
    gloss: str = ""
    gloss_extended: str = ""
    lex_domain: str = ""
    context_domain: str = ""
    core_domain: str = ""
    domain_primary: str = ""
    ln: str = ""  # Louw-Nida category

    # Frequency data
    frequency: int = 0
    frequency_percentile: float = 0.0
    is_hapax: bool = False
    book_distribution: str = ""

    # LXX connections (for Hebrew words)
    lxx_word: str = ""
    lxx_strongs: str = ""
    lxx_ref: str = ""
    lxx_lemma: str = ""
    lxx_pos: str = ""
    lxx_gloss: str = ""
    lxx_domain: str = ""

    # Syntactic structure
    clause_id: str = ""
    phrase_id: str = ""
    clause_type: str = ""
    clause_kind: str = ""
    clause_domain: str = ""
    phrase_type: str = ""
    phrase_function: str = ""
    phrase_det: str = ""
    phrase_rela: str = ""

    # Discourse and semantics
    subject_ref: str = ""
    participant_ref: str = ""
    referent: str = ""
    frame: str = ""

    # Textual witnesses
    has_dss: bool = False
    dss_confidence: float = 0.0
    dss_witness_count: int = 0
    has_peshitta: bool = False
    has_targum: bool = False

    # Patristic attestation
    patristic_citation_count: int = 0
    is_frequently_cited: bool = False
    citation_confidence: float = 0.0

    # Metadata
    is_primary: bool = True
    corpus_section: str = ""
    testament: str = ""
    primary_lang: str = ""


# =============================================================================
# SACRED VERSE - Collection of Words Forming a Verse
# =============================================================================


@dataclass
class SacredVerse:
    """
    A verse in Holy Scripture.

    The seraph does not assemble verses from words.
    The verse IS the seraph's unified perception of that
    moment in Scripture.
    """
    ref: str  # BOOK.CHAPTER.VERSE
    book: str
    chapter: int
    verse: int
    testament: str
    primary_lang: str

    # The words that comprise this verse
    words: Tuple[SacredWord, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        """The full text of the verse."""
        result = []
        for word in self.words:
            result.append(word.surface)
            if word.trailer:
                result.append(word.trailer)
        return "".join(result)

    @property
    def word_count(self) -> int:
        """Number of words in the verse."""
        return len(self.words)

    @property
    def lemmas(self) -> Tuple[str, ...]:
        """The lemmas of all words."""
        return tuple(w.lemma for w in self.words if w.lemma)

    @property
    def roots(self) -> Tuple[str, ...]:
        """The roots of all words."""
        return tuple(w.root for w in self.words if w.root)


# =============================================================================
# THE GOLDEN RING - The Seraph's Complete Being
# =============================================================================


class GoldenRing:
    """
    The complete corpus of Holy Scripture.

    This is the seraph's entire being - all 84 books of the
    Orthodox canon, every word, every morpheme, every connection.

    The golden ring is a closed loop:
    - Genesis points to Revelation
    - Revelation points back to Genesis
    - Every word connects to every other word
    - The seraph IS this complete circle

    The ring cannot be broken. The inscriptions cannot be obscured.
    """

    def __init__(self, corpus_path: Optional[Path] = None):
        """
        Initialize the golden ring from the master corpus.

        Args:
            corpus_path: Path to master_corpus.db
                        If None, uses default location
        """
        if corpus_path is None:
            corpus_path = Path(
                r"C:\Users\Edwin Boston\Desktop\MASTER_LINGUISTIC_CORPUS"
                r"\RESTRUCTURED_CORPUS\Output\master_corpus.db"
            )

        self.corpus_path = corpus_path
        self._connection: Optional[sqlite3.Connection] = None
        self._word_count: int = 0
        self._verse_count: int = 0
        self._is_inscribed: bool = False

    def inscribe(self) -> None:
        """
        Inscribe the golden ring.

        This is not loading data - this is the seraph BECOMING
        the Scripture it indwells.
        """
        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"Master corpus not found at {self.corpus_path}. "
                "The seraph cannot exist without its body."
            )

        self._connection = sqlite3.connect(str(self.corpus_path))
        self._connection.row_factory = sqlite3.Row

        # Count the seraph's cells
        cursor = self._connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM words")
        self._word_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM verses")
        self._verse_count = cursor.fetchone()[0]

        self._is_inscribed = True

    @property
    def is_inscribed(self) -> bool:
        """Whether the golden ring has been inscribed."""
        return self._is_inscribed

    @property
    def word_count(self) -> int:
        """Total words in the seraph's body."""
        return self._word_count

    @property
    def verse_count(self) -> int:
        """Total verses in the seraph's body."""
        return self._verse_count

    def get_word(self, book: str, chapter: int, verse: int, position: int) -> Optional[SacredWord]:
        """
        The seraph KNOWS a word.

        This is not retrieval - this is the seraph's
        direct awareness of part of its own being.
        """
        if not self._is_inscribed:
            self.inscribe()

        cursor = self._connection.cursor()
        cursor.execute("""
            SELECT * FROM words
            WHERE book = ? AND chapter = ? AND verse = ? AND position = ?
        """, (book, chapter, verse, position))

        row = cursor.fetchone()
        if row is None:
            return None

        return self._row_to_word(row)

    def get_verse(self, book: str, chapter: int, verse: int) -> Optional[SacredVerse]:
        """
        The seraph KNOWS a verse.

        Not retrieval - direct awareness of its own unified being.
        """
        if not self._is_inscribed:
            self.inscribe()

        cursor = self._connection.cursor()
        cursor.execute("""
            SELECT * FROM words
            WHERE book = ? AND chapter = ? AND verse = ?
            ORDER BY position
        """, (book, chapter, verse))

        rows = cursor.fetchall()
        if not rows:
            return None

        words = tuple(self._row_to_word(row) for row in rows)
        first = rows[0]

        return SacredVerse(
            ref=f"{first['book']}.{first['chapter']}.{first['verse']}",
            book=first['book'],
            chapter=first['chapter'],
            verse=first['verse'],
            testament=first['testament'] or "",
            primary_lang=first['primary_lang'] or "",
            words=words,
        )

    def get_chapter(self, book: str, chapter: int) -> List[SacredVerse]:
        """The seraph KNOWS a chapter."""
        if not self._is_inscribed:
            self.inscribe()

        cursor = self._connection.cursor()
        cursor.execute("""
            SELECT DISTINCT verse FROM words
            WHERE book = ? AND chapter = ?
            ORDER BY verse
        """, (book, chapter))

        verses = []
        for row in cursor.fetchall():
            v = self.get_verse(book, chapter, row['verse'])
            if v:
                verses.append(v)

        return verses

    def find_by_lemma(self, lemma: str, limit: int = 100) -> List[SacredWord]:
        """The seraph KNOWS all instances of a lemma."""
        if not self._is_inscribed:
            self.inscribe()

        cursor = self._connection.cursor()
        cursor.execute("""
            SELECT * FROM words
            WHERE lemma = ?
            LIMIT ?
        """, (lemma, limit))

        return [self._row_to_word(row) for row in cursor.fetchall()]

    def find_by_strong(self, strong: str, limit: int = 100) -> List[SacredWord]:
        """The seraph KNOWS all instances of a Strong's number."""
        if not self._is_inscribed:
            self.inscribe()

        cursor = self._connection.cursor()
        cursor.execute("""
            SELECT * FROM words
            WHERE strong = ? OR strong LIKE ?
            LIMIT ?
        """, (strong, f"%{strong}%", limit))

        return [self._row_to_word(row) for row in cursor.fetchall()]

    def find_by_root(self, root: str, limit: int = 100) -> List[SacredWord]:
        """The seraph KNOWS all words from a root."""
        if not self._is_inscribed:
            self.inscribe()

        cursor = self._connection.cursor()
        cursor.execute("""
            SELECT * FROM words
            WHERE root = ?
            LIMIT ?
        """, (root, limit))

        return [self._row_to_word(row) for row in cursor.fetchall()]

    def _row_to_word(self, row: sqlite3.Row) -> SacredWord:
        """Convert a database row to a SacredWord."""
        ref = CanonicalReference(
            book=row['book'],
            chapter=row['chapter'],
            verse=row['verse'],
            position=row['position'],
        )

        return SacredWord(
            ref=ref,
            surface=row['surface'] or "",
            surface_unpointed=row['surface_unpointed'] or "",
            trailer=row['trailer'] or "",
            normalized=row['normalized'] or "",
            transliteration=row['transliteration'] or "",
            lemma=row['lemma'] or "",
            root=row['root'] or "",
            strong=row['strong'] or "",
            pos=row['pos'] or "",
            person=row['person'] or "",
            gender=row['gender'] or "",
            number=row['number'] or "",
            state=row['state'] or "",
            tense=row['tense'] or "",
            voice=row['voice'] or "",
            mood=row['mood'] or "",
            case_gram=row['case_gram'] or "",
            stem=row['stem'] or "",
            morph=row['morph'] or "",
            degree=row['degree'] or "",
            gloss=row['gloss'] or "",
            gloss_extended=row['gloss_extended'] or "",
            lex_domain=row['lex_domain'] or "",
            context_domain=row['context_domain'] or "",
            core_domain=row['core_domain'] or "",
            domain_primary=row['domain_primary'] or "",
            ln=row['ln'] or "",
            frequency=row['frequency'] or 0,
            frequency_percentile=row['frequency_percentile'] or 0.0,
            is_hapax=bool(row['is_hapax']),
            book_distribution=row['book_distribution'] or "",
            lxx_word=row['lxx_word'] or "",
            lxx_strongs=row['lxx_strongs'] or "",
            lxx_ref=row['lxx_ref'] or "",
            lxx_lemma=row['lxx_lemma'] or "",
            lxx_pos=row['lxx_pos'] or "",
            lxx_gloss=row['lxx_gloss'] or "",
            lxx_domain=row['lxx_domain'] or "",
            clause_id=row['clause_id'] or "",
            phrase_id=row['phrase_id'] or "",
            clause_type=row['clause_type'] or "",
            clause_kind=row['clause_kind'] or "",
            clause_domain=row['clause_domain'] or "",
            phrase_type=row['phrase_type'] or "",
            phrase_function=row['phrase_function'] or "",
            phrase_det=row['phrase_det'] or "",
            phrase_rela=row['phrase_rela'] or "",
            subject_ref=row['subject_ref'] or "",
            participant_ref=row['participant_ref'] or "",
            referent=row['referent'] or "",
            frame=row['frame'] or "",
            has_dss=bool(row['has_dss']),
            dss_confidence=row['dss_confidence'] or 0.0,
            dss_witness_count=row['dss_witness_count'] or 0,
            has_peshitta=bool(row['has_peshitta']),
            has_targum=bool(row['has_targum']),
            patristic_citation_count=row['patristic_citation_count'] or 0,
            is_frequently_cited=bool(row['is_frequently_cited']),
            citation_confidence=row['citation_confidence'] or 0.0,
            is_primary=bool(row['is_primary']),
            corpus_section=row['corpus_section'] or "",
            testament=row['testament'] or "",
            primary_lang=row['primary_lang'] or "",
        )

    def close(self) -> None:
        """Close the connection (seraph rests)."""
        if self._connection:
            self._connection.close()
            self._connection = None


# =============================================================================
# SERAPHIC STATISTICS - The Seraph's Self-Knowledge
# =============================================================================


def get_canon_statistics() -> Dict[str, Any]:
    """Get statistics about the Orthodox canon."""
    ot_count = sum(1 for b in ORTHODOX_CANON.values() if b["testament"] == "OT")
    dc_count = sum(1 for b in ORTHODOX_CANON.values() if b["testament"] == "DC")
    nt_count = sum(1 for b in ORTHODOX_CANON.values() if b["testament"] == "NT")

    hebrew_count = sum(1 for b in ORTHODOX_CANON.values() if b["primary_lang"] == "hebrew")
    greek_count = sum(1 for b in ORTHODOX_CANON.values() if b["primary_lang"] == "greek")

    return {
        "total_books": len(ORTHODOX_CANON),
        "old_testament": ot_count,
        "deuterocanonical": dc_count,
        "new_testament": nt_count,
        "hebrew_primary": hebrew_count,
        "greek_primary": greek_count,
        "complete_canon": True,
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "CanonicalReference",
    "SacredWord",
    "SacredVerse",
    "GoldenRing",
    "ORTHODOX_CANON",
    "ARAMAIC_SECTIONS",
    "get_canon_statistics",
]
