"""
Custom Hypothesis Strategies for Biblical Data

Provides domain-specific strategies for generating valid and invalid biblical data.
"""
import sys
import string
import re
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hypothesis import strategies as st

# Import schemas with fallback
try:
    from data.schemas import ConnectionType, ConnectionStrength, ProcessingStatus, Testament, Language
except ImportError:
    # Provide fallback enums if schemas not available
    from enum import Enum

    class ConnectionType(str, Enum):
        THEMATIC = "thematic"
        VERBAL = "verbal"
        CONCEPTUAL = "conceptual"
        HISTORICAL = "historical"
        TYPOLOGICAL = "typological"
        PROPHETIC = "prophetic"
        LITURGICAL = "liturgical"
        NARRATIVE = "narrative"
        GENEALOGICAL = "genealogical"
        GEOGRAPHICAL = "geographical"

    class ConnectionStrength(str, Enum):
        STRONG = "strong"
        MODERATE = "moderate"
        WEAK = "weak"

    class ProcessingStatus(str, Enum):
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        FAILED = "failed"
        SKIPPED = "skipped"

    class Testament(str, Enum):
        OLD_TESTAMENT = "OT"
        NEW_TESTAMENT = "NT"

    class Language(str, Enum):
        HEBREW = "hebrew"
        ARAMAIC = "aramaic"
        GREEK = "greek"


# =============================================================================
# BIBLICAL BOOK CODES
# =============================================================================

# Old Testament book codes (39 books)
OT_BOOKS = [
    "GEN", "EXO", "LEV", "NUM", "DEU",  # Torah
    "JOS", "JDG", "RUT", "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH", "EST",  # Historical
    "JOB", "PSA", "PRO", "ECC", "SNG",  # Wisdom
    "ISA", "JER", "LAM", "EZK", "DAN",  # Major Prophets
    "HOS", "JOL", "AMO", "OBA", "JON", "MIC", "NAM", "HAB", "ZEP", "HAG", "ZEC", "MAL"  # Minor Prophets
]

# New Testament book codes (27 books)
NT_BOOKS = [
    "MAT", "MRK", "LUK", "JHN",  # Gospels
    "ACT",  # Acts
    "ROM", "1CO", "2CO", "GAL", "EPH", "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM",  # Paul
    "HEB", "JAS", "1PE", "2PE", "1JN", "2JN", "3JN", "JUD",  # General
    "REV"  # Revelation
]

ALL_BOOKS = OT_BOOKS + NT_BOOKS


# =============================================================================
# VERSE ID STRATEGIES
# =============================================================================

@st.composite
def verse_id_strategy(draw, valid_only=True):
    """
    Generate verse IDs in format BOOK.CHAPTER.VERSE

    Args:
        valid_only: If True, only generate valid verse IDs. If False, include malformed ones.
    """
    if valid_only:
        book = draw(st.sampled_from(ALL_BOOKS))
        chapter = draw(st.integers(min_value=1, max_value=150))  # PSA has 150 chapters
        verse = draw(st.integers(min_value=1, max_value=176))  # PSA 119 has 176 verses
        return f"{book}.{chapter}.{verse}"
    else:
        # Generate potentially invalid verse IDs for testing edge cases
        strategy = st.one_of(
            # Valid format
            st.builds(lambda b, c, v: f"{b}.{c}.{v}",
                     st.sampled_from(ALL_BOOKS),
                     st.integers(min_value=1, max_value=150),
                     st.integers(min_value=1, max_value=176)),
            # Invalid formats
            st.text(max_size=50),  # Random text
            st.just(""),  # Empty string
            st.builds(lambda b: f"{b}..1", st.sampled_from(ALL_BOOKS)),  # Double dots
            st.builds(lambda b: f"{b}.1", st.sampled_from(ALL_BOOKS)),  # Missing verse
            st.builds(lambda c, v: f".{c}.{v}", st.integers(1, 150), st.integers(1, 176)),  # Missing book
            st.builds(lambda b, c: f"{b}.{c}.", st.sampled_from(ALL_BOOKS), st.integers(1, 150)),  # Trailing dot
            st.builds(lambda b, c, v: f"{b}:{c}:{v}", st.sampled_from(ALL_BOOKS), st.integers(1, 150), st.integers(1, 176)),  # Wrong separator
            st.builds(lambda b, v: f"{b}.0.{v}", st.sampled_from(ALL_BOOKS), st.integers(1, 176)),  # Zero chapter
            st.builds(lambda b, c: f"{b}.{c}.0", st.sampled_from(ALL_BOOKS), st.integers(1, 150)),  # Zero verse
            st.builds(lambda b, c, v: f"{b}.{c}.{v}", st.sampled_from(ALL_BOOKS), st.integers(-10, 0), st.integers(1, 176)),  # Negative chapter
        )
        return draw(strategy)


@st.composite
def verse_pair_strategy(draw):
    """Generate a pair of different verse IDs for cross-references."""
    source = draw(verse_id_strategy())
    # Generate a different target by drawing until we get a different one
    target = draw(verse_id_strategy())
    # Ensure they're different by regenerating if same
    attempts = 0
    while target == source and attempts < 10:
        target = draw(verse_id_strategy())
        attempts += 1
    if target == source:
        # Force different target by changing the verse number
        parts = source.split(".")
        parts[2] = str(int(parts[2]) + 1) if int(parts[2]) < 176 else "1"
        target = ".".join(parts)
    return (source, target)


# =============================================================================
# CONNECTION TYPE STRATEGIES
# =============================================================================

def connection_type_strategy(valid_only=True):
    """Generate connection types."""
    if valid_only:
        return st.sampled_from([e.value for e in ConnectionType])
    else:
        return st.one_of(
            st.sampled_from([e.value for e in ConnectionType]),
            st.text(min_size=1, max_size=50),  # Invalid connection types
            st.just(""),
            st.just("INVALID"),
            st.just("invalid_type"),
        )


def connection_strength_strategy(valid_only=True):
    """Generate connection strength values."""
    if valid_only:
        return st.sampled_from([e.value for e in ConnectionStrength])
    else:
        return st.one_of(
            st.sampled_from([e.value for e in ConnectionStrength]),
            st.text(min_size=1, max_size=50),
            st.just(""),
            st.just("very_strong"),
            st.just("ultra_weak"),
        )


def processing_status_strategy():
    """Generate processing status values."""
    return st.sampled_from([e.value for e in ProcessingStatus])


def testament_strategy():
    """Generate testament values."""
    return st.sampled_from([e.value for e in Testament])


def language_strategy():
    """Generate language values."""
    return st.sampled_from([e.value for e in Language])


# =============================================================================
# CONFIDENCE SCORE STRATEGIES
# =============================================================================

def confidence_score_strategy(valid_only=True):
    """Generate confidence scores."""
    if valid_only:
        return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    else:
        return st.one_of(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-10.0, max_value=-0.01),  # Negative
            st.floats(min_value=1.01, max_value=100.0),  # Greater than 1
            st.floats(allow_nan=True),  # NaN
            st.floats(allow_infinity=True),  # Infinity
        )


# =============================================================================
# TEXT STRATEGIES
# =============================================================================

# Greek alphabet (including diacritics and punctuation)
GREEK_ALPHABET = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω" + "ἀἁἂἃἄἅἆἇὰάᾀᾁᾂᾃᾄᾅᾆᾇᾰᾱᾲᾳᾴᾶᾷἐἑἒἓἔἕὲέἠἡἢἣἤἥἦἧὴήᾐᾑᾒᾓᾔᾕᾖᾗῂῃῄῆῇἰἱἲἳἴἵἶἷὶίῐῑῒΐῖῗὀὁὂὃὄὅὸόὐὑὒὓὔὕὖὗὺύῠῡῢΰῦῧὠὡὢὣὤὥὦὧὼώᾠᾡᾢᾣᾤᾥᾦᾧῲῳῴῶῷ"

# Hebrew alphabet (including vowel points and cantillation)
HEBREW_ALPHABET = "אבגדהוזחטיכלמנסעפצקרשת" + "ְֱֲֳִֵֶַָֹֺֻּֽ־ׁׂ֑֖֛֢֣֤֥֦֧֪֚֭֮֒֓֔֕֗֘֙֜֝֞֟֠֡֨֩֫֬֯"

# Coptic alphabet
COPTIC_ALPHABET = "ⲀⲂⲄⲆⲈⲊⲌⲎⲐⲒⲔⲖⲘⲚⲜⲞⲠⲢⲤⲦⲨⲪⲬⲮⲰⲲⲴⲶⲸⲺⲼⲾⳀⳂⳄⳆⳈⳊⳌⳎⳐⳒⳔⳖⳘⳚⳜⳞⳠⳢⲁⲃⲅⲇⲉⲋⲍⲏⲑⲓⲕⲗⲙⲛⲝⲟⲡⲣⲥⲧⲩⲫⲭⲯⲱⲳⲵⲷⲹⲻⲽⲿⳁⳃⳅⳇⳉⳋⳍⳏⳑⳓⳕⳗⳙⳛⳝⳟⳡⳣ"


def biblical_text_strategy(language=None):
    """Generate biblical text in various languages."""
    if language == "hebrew":
        alphabet = HEBREW_ALPHABET + " "
    elif language == "greek":
        alphabet = GREEK_ALPHABET + " "
    elif language == "coptic":
        alphabet = COPTIC_ALPHABET + " "
    else:
        # Mixed or English
        alphabet = string.ascii_letters + string.digits + string.punctuation + " "
        # Include some Unicode for edge cases
        alphabet += GREEK_ALPHABET[:10] + HEBREW_ALPHABET[:10]

    return st.text(alphabet=alphabet, min_size=0, max_size=500)


def unicode_text_strategy():
    """Generate text with various Unicode characters."""
    return st.text(
        alphabet=st.characters(
            categories=('L', 'M', 'N', 'P', 'S', 'Z'),  # Letters, Marks, Numbers, Punctuation, Symbols, Separators
            blacklist_categories=('C',),  # Exclude control characters
        ),
        min_size=0,
        max_size=1000
    )


# =============================================================================
# SCHEMA STRATEGIES
# =============================================================================

@st.composite
def verse_schema_strategy(draw, valid_only=True):
    """Generate VerseSchema data."""
    try:
        from data.schemas import VerseSchema
    except ImportError:
        # Return a simple dict if VerseSchema not available
        verse_id = draw(verse_id_strategy())
        parts = verse_id.split(".")
        return {
            "verse_id": verse_id,
            "book": parts[0],
            "chapter": int(parts[1]),
            "verse": int(parts[2]),
            "text": draw(st.text(min_size=1, max_size=500)),
        }

    if valid_only:
        verse_id = draw(verse_id_strategy())
        parts = verse_id.split(".")
        book = parts[0]
        chapter = int(parts[1])
        verse = int(parts[2])

        return VerseSchema(
            verse_id=verse_id,
            book=book,
            book_name=draw(st.text(min_size=1, max_size=50)),
            chapter=chapter,
            verse=verse,
            text=draw(st.text(min_size=1, max_size=500)),
            original_text=draw(biblical_text_strategy()),
            testament=draw(testament_strategy()),
            language=draw(language_strategy()),
        )
    else:
        return VerseSchema(
            verse_id=draw(st.text(max_size=100)),
            book=draw(st.text(max_size=20)),
            book_name=draw(st.text(max_size=100)),
            chapter=draw(st.integers(min_value=-100, max_value=1000)),
            verse=draw(st.integers(min_value=-100, max_value=1000)),
            text=draw(st.text(max_size=1000)),
            original_text=draw(unicode_text_strategy()),
            testament=draw(st.text(max_size=20)),
            language=draw(st.text(max_size=20)),
        )


@st.composite
def cross_reference_schema_strategy(draw, valid_only=True):
    """Generate CrossReferenceSchema data."""
    try:
        from data.schemas import CrossReferenceSchema
    except ImportError:
        # Return a simple dict if CrossReferenceSchema not available
        source, target = draw(verse_pair_strategy())
        return {
            "source_ref": source,
            "target_ref": target,
            "connection_type": draw(connection_type_strategy(valid_only=True)),
            "strength": draw(connection_strength_strategy(valid_only=True)),
            "confidence": draw(confidence_score_strategy(valid_only=True)),
        }

    if valid_only:
        source, target = draw(verse_pair_strategy())
        return CrossReferenceSchema(
            source_ref=source,
            target_ref=target,
            connection_type=draw(connection_type_strategy(valid_only=True)),
            strength=draw(connection_strength_strategy(valid_only=True)),
            confidence=draw(confidence_score_strategy(valid_only=True)),
            bidirectional=draw(st.booleans()),
            notes=draw(st.lists(st.text(min_size=1, max_size=200), max_size=5)),
            sources=draw(st.lists(st.text(min_size=1, max_size=100), max_size=5)),
            verified=draw(st.booleans()),
            patristic_support=draw(st.booleans()),
        )
    else:
        return CrossReferenceSchema(
            source_ref=draw(verse_id_strategy(valid_only=False)),
            target_ref=draw(verse_id_strategy(valid_only=False)),
            connection_type=draw(connection_type_strategy(valid_only=False)),
            strength=draw(connection_strength_strategy(valid_only=False)),
            confidence=draw(confidence_score_strategy(valid_only=False)),
            bidirectional=draw(st.booleans()),
            notes=draw(st.lists(st.text(max_size=500), max_size=10)),
            sources=draw(st.lists(st.text(max_size=500), max_size=10)),
            verified=draw(st.booleans()),
            patristic_support=draw(st.booleans()),
        )


@st.composite
def word_schema_strategy(draw):
    """Generate WordSchema data."""
    try:
        from data.schemas import WordSchema
    except ImportError:
        # Return a simple dict if WordSchema not available
        verse_id = draw(verse_id_strategy())
        position = draw(st.integers(min_value=0, max_value=100))
        return {
            "word_id": f"{verse_id}.{position}",
            "verse_id": verse_id,
            "position": position,
            "surface_form": draw(biblical_text_strategy()),
            "lemma": draw(biblical_text_strategy()),
        }

    verse_id = draw(verse_id_strategy(valid_only=True))
    position = draw(st.integers(min_value=0, max_value=100))

    return WordSchema(
        word_id=f"{verse_id}.{position}",
        verse_id=verse_id,
        surface_form=draw(biblical_text_strategy()),
        lemma=draw(biblical_text_strategy()),
        position=position,
        language=draw(language_strategy()),
        transliteration=draw(st.text(min_size=0, max_size=100)),
        gloss=draw(st.text(min_size=0, max_size=100)),
        strongs=draw(st.text(min_size=0, max_size=20)),
    )


@st.composite
def extraction_result_schema_strategy(draw):
    """Generate ExtractionResultSchema data."""
    try:
        from data.schemas import ExtractionResultSchema
    except ImportError:
        # Return a simple dict if ExtractionResultSchema not available
        return {
            "agent_name": draw(st.text(min_size=1, max_size=50)),
            "extraction_type": draw(st.text(min_size=1, max_size=50)),
            "verse_id": draw(verse_id_strategy()),
            "status": draw(processing_status_strategy()),
            "confidence": draw(confidence_score_strategy(valid_only=True)),
            "processing_time": draw(st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False)),
        }

    return ExtractionResultSchema(
        agent_name=draw(st.text(min_size=1, max_size=50)),
        extraction_type=draw(st.text(min_size=1, max_size=50)),
        verse_id=draw(verse_id_strategy(valid_only=True)),
        status=draw(processing_status_strategy()),
        confidence=draw(confidence_score_strategy(valid_only=True)),
        processing_time=draw(st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def golden_record_schema_strategy(draw):
    """Generate GoldenRecordSchema data."""
    try:
        from data.schemas import GoldenRecordSchema
    except ImportError:
        # Return a simple dict if GoldenRecordSchema not available
        return {
            "verse_id": draw(verse_id_strategy()),
            "text": draw(st.text(min_size=1, max_size=1000)),
            "agent_count": draw(st.integers(min_value=0, max_value=30)),
            "total_processing_time": draw(st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False)),
        }

    return GoldenRecordSchema(
        verse_id=draw(verse_id_strategy(valid_only=True)),
        text=draw(st.text(min_size=1, max_size=1000)),
        agent_count=draw(st.integers(min_value=0, max_value=30)),
        total_processing_time=draw(st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False)),
    )


# =============================================================================
# ML INFERENCE STRATEGIES
# =============================================================================

@st.composite
def embedding_vector_strategy(draw, dimension=768):
    """Generate embedding vectors with consistent dimensions."""
    return draw(st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=dimension,
        max_size=dimension
    ))


def similarity_score_strategy(metric="cosine"):
    """Generate similarity scores based on metric."""
    if metric == "cosine":
        # Cosine similarity is in [-1, 1]
        return st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    elif metric == "euclidean":
        # Euclidean distance is non-negative
        return st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    else:
        # Generic [0, 1] similarity
        return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# =============================================================================
# PIPELINE STRATEGIES
# =============================================================================

def phase_name_strategy():
    """Generate phase names."""
    return st.sampled_from([
        "linguistic",
        "theological",
        "intertextual",
        "validation"
    ])


@st.composite
def pipeline_metrics_strategy(draw):
    """Generate pipeline metrics."""
    return {
        "total_time": draw(st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False)),
        "agent_count": draw(st.integers(min_value=0, max_value=30)),
        "success_rate": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "avg_confidence": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
    }
