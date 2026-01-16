"""
BIBLOS v2 - Text-Fabric Integration

Integration with Text-Fabric for BHSA (Hebrew) and Greek NT corpora.
"""
import asyncio
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


class TextFabricIntegration(BaseCorpusIntegration):
    """
    Text-Fabric corpus integration.

    Provides access to:
    - BHSA: Biblia Hebraica Stuttgartensia Amstelodamensis
    - Greek NT corpora (SBLGNT, NA28)
    """

    # Hebrew OT books
    OT_BOOKS = [
        "GEN", "EXO", "LEV", "NUM", "DEU",
        "JOS", "JDG", "RUT", "1SA", "2SA",
        "1KI", "2KI", "1CH", "2CH", "EZR",
        "NEH", "EST", "JOB", "PSA", "PRO",
        "ECC", "SNG", "ISA", "JER", "LAM",
        "EZK", "DAN", "HOS", "JOL", "AMO",
        "OBA", "JON", "MIC", "NAH", "HAB",
        "ZEP", "HAG", "ZEC", "MAL"
    ]

    # Greek NT books
    NT_BOOKS = [
        "MAT", "MRK", "LUK", "JHN", "ACT",
        "ROM", "1CO", "2CO", "GAL", "EPH",
        "PHP", "COL", "1TH", "2TH", "1TI",
        "2TI", "TIT", "PHM", "HEB", "JAS",
        "1PE", "2PE", "1JN", "2JN", "3JN",
        "JUD", "REV"
    ]

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        corpus_type: str = "bhsa"
    ):
        super().__init__(corpus_path)
        self.corpus_type = corpus_type
        self._tf = None
        self._api = None
        self._features = {}

    async def initialize(self) -> None:
        """Initialize Text-Fabric connection."""
        self.logger.info(f"Initializing Text-Fabric integration ({self.corpus_type})")

        try:
            # Import Text-Fabric
            from tf.app import use

            # Load corpus
            if self.corpus_type == "bhsa":
                self._tf = use("ETCBC/bhsa", silent="deep")
            elif self.corpus_type == "sblgnt":
                self._tf = use("ETCBC/sblgnt", silent="deep")
            else:
                self._tf = use(self.corpus_type, silent="deep")

            self._api = self._tf.api
            self._initialized = True
            self.logger.info("Text-Fabric initialized successfully")

        except ImportError:
            self.logger.warning("Text-Fabric not installed, using mock mode")
            self._initialized = True  # Allow mock mode
        except Exception as e:
            self.logger.error(f"Failed to initialize Text-Fabric: {e}")
            self._initialized = True  # Allow degraded mode

    async def get_verse(self, verse_id: str) -> Optional[VerseData]:
        """Get verse data by ID."""
        if not self._initialized:
            await self.initialize()

        parsed = self._parse_verse_id(verse_id)
        book = self._normalize_book_code(parsed["book"])
        chapter = parsed["chapter"]
        verse = parsed["verse"]

        # Determine language
        language = Language.HEBREW if book in self.OT_BOOKS else Language.GREEK
        text_type = TextType.MASORETIC if language == Language.HEBREW else TextType.SBLGNT

        # If Text-Fabric loaded, use it
        if self._api:
            try:
                verse_data = await self._get_tf_verse(book, chapter, verse)
                if verse_data:
                    return verse_data
            except Exception as e:
                self.logger.warning(f"Text-Fabric lookup failed: {e}")

        # Return mock/placeholder data
        return VerseData(
            verse_id=f"{book}.{chapter}.{verse}",
            book=book,
            chapter=chapter,
            verse=verse,
            text=f"[{book} {chapter}:{verse}]",
            language=language,
            text_type=text_type,
            words=[],
            metadata={"source": "mock"}
        )

    async def _get_tf_verse(
        self,
        book: str,
        chapter: int,
        verse: int
    ) -> Optional[VerseData]:
        """Get verse using Text-Fabric API."""
        if not self._api:
            return None

        T = self._api.T
        F = self._api.F
        L = self._api.L

        # Find verse node
        verse_nodes = T.nodeFromSection((book, chapter, verse))
        if not verse_nodes:
            return None

        verse_node = verse_nodes[0] if isinstance(verse_nodes, tuple) else verse_nodes

        # Get text
        text = T.text(verse_node)

        # Get words
        words = []
        word_nodes = L.d(verse_node, "word")

        for pos, word_node in enumerate(word_nodes):
            word_data = await self._get_tf_word(word_node, pos)
            if word_data:
                words.append(word_data)

        # Determine language
        is_hebrew = book in self.OT_BOOKS
        language = Language.HEBREW if is_hebrew else Language.GREEK
        text_type = TextType.MASORETIC if is_hebrew else TextType.SBLGNT

        return VerseData(
            verse_id=f"{book}.{chapter}.{verse}",
            book=book,
            chapter=chapter,
            verse=verse,
            text=text,
            language=language,
            text_type=text_type,
            words=words,
            metadata={"source": "text-fabric", "corpus": self.corpus_type}
        )

    async def _get_tf_word(
        self,
        word_node: int,
        position: int
    ) -> Optional[WordData]:
        """Get word data from Text-Fabric node."""
        if not self._api:
            return None

        F = self._api.F

        # Get surface form
        surface = F.g_word_utf8.v(word_node) if hasattr(F, "g_word_utf8") else F.g_word.v(word_node)
        if not surface:
            surface = str(word_node)

        # Get lemma
        lemma = F.lex_utf8.v(word_node) if hasattr(F, "lex_utf8") else F.lex.v(word_node)
        if not lemma:
            lemma = surface

        # Get morphology
        morph = self._parse_tf_morphology(word_node)

        # Get Strong's number
        strongs = F.strongs.v(word_node) if hasattr(F, "strongs") else None

        # Get gloss
        gloss = F.gloss.v(word_node) if hasattr(F, "gloss") else None

        # Determine language from POS
        pos = morph.part_of_speech
        language = Language.HEBREW  # Default, would check corpus type

        return WordData(
            word_id=str(word_node),
            surface_form=surface,
            lemma=lemma,
            language=language,
            morphology=morph,
            position=position,
            transliteration=F.g_word.v(word_node) if hasattr(F, "g_word") else None,
            gloss=gloss,
            strongs=strongs,
            syntax_role=F.function.v(word_node) if hasattr(F, "function") else None
        )

    def _parse_tf_morphology(self, word_node: int) -> MorphologyData:
        """Parse morphology from Text-Fabric features."""
        if not self._api:
            return MorphologyData(part_of_speech="unknown")

        F = self._api.F

        # Hebrew morphology features
        pos = F.sp.v(word_node) if hasattr(F, "sp") else "unknown"
        person = F.ps.v(word_node) if hasattr(F, "ps") else None
        number = F.nu.v(word_node) if hasattr(F, "nu") else None
        gender = F.gn.v(word_node) if hasattr(F, "gn") else None
        state = F.st.v(word_node) if hasattr(F, "st") else None
        stem = F.vs.v(word_node) if hasattr(F, "vs") else None
        tense = F.vt.v(word_node) if hasattr(F, "vt") else None

        return MorphologyData(
            part_of_speech=pos or "unknown",
            person=person,
            number=number,
            gender=gender,
            state=state,
            stem=stem,
            tense=tense
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
        verses = []

        if self._api:
            try:
                T = self._api.T

                if chapter:
                    # Get specific chapter
                    chapter_node = T.nodeFromSection((book, chapter))
                    if chapter_node:
                        verse_nodes = self._api.L.d(chapter_node, "verse")
                        for v_node in verse_nodes:
                            section = T.sectionFromNode(v_node)
                            verse_data = await self.get_verse(
                                f"{section[0]}.{section[1]}.{section[2]}"
                            )
                            if verse_data:
                                verses.append(verse_data)
                else:
                    # Get all chapters in book
                    book_node = T.nodeFromSection((book,))
                    if book_node:
                        verse_nodes = self._api.L.d(book_node, "verse")
                        for v_node in verse_nodes:
                            section = T.sectionFromNode(v_node)
                            verse_data = await self.get_verse(
                                f"{section[0]}.{section[1]}.{section[2]}"
                            )
                            if verse_data:
                                verses.append(verse_data)
            except Exception as e:
                self.logger.error(f"Failed to get verses: {e}")

        return verses

    async def search(
        self,
        query: str,
        **kwargs
    ) -> List[VerseData]:
        """Search the corpus using Text-Fabric query language."""
        if not self._initialized:
            await self.initialize()

        results = []

        if self._api:
            try:
                S = self._api.S
                T = self._api.T

                # Execute TF search
                search_results = S.search(query)

                for result in search_results[:kwargs.get("limit", 100)]:
                    # Get verse from result
                    node = result[0] if isinstance(result, tuple) else result
                    verse_node = self._api.L.u(node, "verse")

                    if verse_node:
                        section = T.sectionFromNode(verse_node[0])
                        verse_data = await self.get_verse(
                            f"{section[0]}.{section[1]}.{section[2]}"
                        )
                        if verse_data and verse_data not in results:
                            results.append(verse_data)
            except Exception as e:
                self.logger.error(f"Search failed: {e}")

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
        if self.corpus_type in ["bhsa", "hebrew"]:
            return self.OT_BOOKS
        elif self.corpus_type in ["sblgnt", "greek"]:
            return self.NT_BOOKS
        return self.OT_BOOKS + self.NT_BOOKS

    def get_language(self) -> Language:
        """Get the primary language of this corpus."""
        if self.corpus_type in ["bhsa", "hebrew"]:
            return Language.HEBREW
        return Language.GREEK

    @lru_cache(maxsize=1000)
    def _cache_verse(self, verse_id: str) -> Dict[str, Any]:
        """Cache verse lookups."""
        # This is a sync helper for caching
        return {}

    async def get_syntax_tree(
        self,
        verse_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get syntactic tree for a verse."""
        if not self._api:
            return None

        parsed = self._parse_verse_id(verse_id)
        book = self._normalize_book_code(parsed["book"])

        try:
            T = self._api.T
            L = self._api.L
            F = self._api.F

            verse_node = T.nodeFromSection((book, parsed["chapter"], parsed["verse"]))
            if not verse_node:
                return None

            # Build clause hierarchy
            clauses = L.d(verse_node, "clause")
            tree = {
                "verse_id": verse_id,
                "clauses": []
            }

            for clause in clauses:
                clause_data = {
                    "clause_id": str(clause),
                    "type": F.typ.v(clause) if hasattr(F, "typ") else "unknown",
                    "phrases": []
                }

                phrases = L.d(clause, "phrase")
                for phrase in phrases:
                    phrase_data = {
                        "phrase_id": str(phrase),
                        "function": F.function.v(phrase) if hasattr(F, "function") else "unknown",
                        "type": F.typ.v(phrase) if hasattr(F, "typ") else "unknown"
                    }
                    clause_data["phrases"].append(phrase_data)

                tree["clauses"].append(clause_data)

            return tree

        except Exception as e:
            self.logger.error(f"Failed to get syntax tree: {e}")
            return None

    async def cleanup(self) -> None:
        """Cleanup Text-Fabric resources."""
        self._tf = None
        self._api = None
        self._initialized = False
        self.logger.info("Text-Fabric integration cleaned up")
