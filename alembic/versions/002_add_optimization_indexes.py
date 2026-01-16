"""Add optimization indexes for BIBLOS v2

Revision ID: 002
Revises: 001
Create Date: 2026-01-15

This migration adds missing indexes identified in the database optimization review:
1. Composite indexes for frequent query patterns
2. GIN indexes for JSONB columns
3. HNSW index for vector similarity search
4. Partial indexes for filtered queries
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add optimization indexes."""

    # =========================================================================
    # Books table indexes
    # =========================================================================
    op.create_index(
        'ix_books_testament',
        'books',
        ['testament'],
        if_not_exists=True
    )
    op.create_index(
        'ix_books_order_num',
        'books',
        ['order_num'],
        if_not_exists=True
    )
    op.create_index(
        'ix_books_testament_order',
        'books',
        ['testament', 'order_num'],
        if_not_exists=True
    )

    # =========================================================================
    # Verses table indexes
    # =========================================================================

    # Full verse location lookup (most specific)
    op.create_index(
        'ix_verses_book_chapter_verse',
        'verses',
        ['book_id', 'chapter', 'verse'],
        if_not_exists=True
    )

    # GIN indexes for JSONB columns (allows querying inside JSON)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_verses_morphology_gin
        ON verses USING gin (morphology)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_verses_syntax_gin
        ON verses USING gin (syntax)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_verses_semantics_gin
        ON verses USING gin (semantics)
    """)

    # HNSW index for vector similarity search
    # Using CONCURRENTLY to avoid locking the table
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_verses_embedding_hnsw
        ON verses USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # =========================================================================
    # Cross-references table indexes
    # =========================================================================

    # Confidence-based filtering by type (high priority queries)
    op.create_index(
        'ix_crossref_confidence_type',
        'cross_references',
        ['confidence', 'connection_type'],
        if_not_exists=True
    )

    # Strength filtering
    op.create_index(
        'ix_crossref_strength',
        'cross_references',
        ['strength'],
        if_not_exists=True
    )

    # Combined strength and confidence for quality filtering
    op.create_index(
        'ix_crossref_strength_confidence',
        'cross_references',
        ['strength', 'confidence'],
        if_not_exists=True
    )

    # Bidirectional lookup optimization
    op.create_index(
        'ix_crossref_source_target',
        'cross_references',
        ['source_id', 'target_id'],
        if_not_exists=True
    )

    # Partial index for high-confidence cross-references only
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_crossref_high_confidence
        ON cross_references (source_id, target_id, connection_type)
        WHERE confidence >= 0.8
    """)

    # Partial index for typological connections (frequently queried)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_crossref_typological
        ON cross_references (source_id, target_id)
        WHERE connection_type = 'typological'
    """)

    # Partial index for prophetic connections (frequently queried)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_crossref_prophetic
        ON cross_references (source_id, target_id)
        WHERE connection_type = 'prophetic'
    """)

    # =========================================================================
    # Patristic citations table indexes
    # =========================================================================

    # Composite index for verse + father lookup
    op.create_index(
        'ix_patristic_verse_father',
        'patristic_citations',
        ['verse_id', 'father_name'],
        if_not_exists=True
    )

    # Century range queries
    op.create_index(
        'ix_patristic_century',
        'patristic_citations',
        ['century'],
        if_not_exists=True
    )

    # Tradition filtering (Greek, Latin, Syriac)
    op.execute("""
        ALTER TABLE patristic_citations
        ADD COLUMN IF NOT EXISTS tradition VARCHAR(50)
    """)
    op.create_index(
        'ix_patristic_tradition',
        'patristic_citations',
        ['tradition'],
        if_not_exists=True
    )

    # Category filtering
    op.create_index(
        'ix_patristic_category',
        'patristic_citations',
        ['category'],
        if_not_exists=True
    )

    # Combined tradition and century for historical queries
    op.create_index(
        'ix_patristic_tradition_century',
        'patristic_citations',
        ['tradition', 'century'],
        if_not_exists=True
    )

    # Partial index for exegetical citations (frequently queried)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_patristic_exegetical
        ON patristic_citations (verse_id, father_name)
        WHERE category = 'exegetical'
    """)

    # =========================================================================
    # Extraction results table indexes
    # =========================================================================

    # Status monitoring
    op.create_index(
        'ix_extraction_status',
        'extraction_results',
        ['status'],
        if_not_exists=True
    )

    # Agent + status for pipeline monitoring
    op.create_index(
        'ix_extraction_agent_status',
        'extraction_results',
        ['agent_name', 'status'],
        if_not_exists=True
    )

    # Extraction type filtering
    op.create_index(
        'ix_extraction_type',
        'extraction_results',
        ['extraction_type'],
        if_not_exists=True
    )

    # Type + status for pipeline monitoring
    op.create_index(
        'ix_extraction_type_status',
        'extraction_results',
        ['extraction_type', 'status'],
        if_not_exists=True
    )

    # Confidence threshold queries
    op.create_index(
        'ix_extraction_confidence',
        'extraction_results',
        ['confidence'],
        if_not_exists=True
    )

    # Validation filtering
    op.execute("""
        ALTER TABLE extraction_results
        ADD COLUMN IF NOT EXISTS validated BOOLEAN DEFAULT FALSE
    """)
    op.create_index(
        'ix_extraction_validated',
        'extraction_results',
        ['validated'],
        if_not_exists=True
    )

    # Partial index for pending extractions (hot path)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_extraction_pending
        ON extraction_results (verse_id, agent_name)
        WHERE status = 'pending'
    """)

    # Partial index for failed extractions (for retry logic)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_extraction_failed
        ON extraction_results (verse_id, agent_name, created_at)
        WHERE status = 'failed'
    """)

    # =========================================================================
    # Update table statistics for query planner
    # =========================================================================
    op.execute("ANALYZE books")
    op.execute("ANALYZE verses")
    op.execute("ANALYZE cross_references")
    op.execute("ANALYZE patristic_citations")
    op.execute("ANALYZE extraction_results")


def downgrade() -> None:
    """Remove optimization indexes."""

    # Books indexes
    op.drop_index('ix_books_testament', if_exists=True)
    op.drop_index('ix_books_order_num', if_exists=True)
    op.drop_index('ix_books_testament_order', if_exists=True)

    # Verses indexes
    op.drop_index('ix_verses_book_chapter_verse', if_exists=True)
    op.execute("DROP INDEX IF EXISTS ix_verses_morphology_gin")
    op.execute("DROP INDEX IF EXISTS ix_verses_syntax_gin")
    op.execute("DROP INDEX IF EXISTS ix_verses_semantics_gin")
    op.execute("DROP INDEX IF EXISTS ix_verses_embedding_hnsw")

    # Cross-references indexes
    op.drop_index('ix_crossref_confidence_type', if_exists=True)
    op.drop_index('ix_crossref_strength', if_exists=True)
    op.drop_index('ix_crossref_strength_confidence', if_exists=True)
    op.drop_index('ix_crossref_source_target', if_exists=True)
    op.execute("DROP INDEX IF EXISTS ix_crossref_high_confidence")
    op.execute("DROP INDEX IF EXISTS ix_crossref_typological")
    op.execute("DROP INDEX IF EXISTS ix_crossref_prophetic")

    # Patristic citations indexes
    op.drop_index('ix_patristic_verse_father', if_exists=True)
    op.drop_index('ix_patristic_century', if_exists=True)
    op.drop_index('ix_patristic_tradition', if_exists=True)
    op.drop_index('ix_patristic_category', if_exists=True)
    op.drop_index('ix_patristic_tradition_century', if_exists=True)
    op.execute("DROP INDEX IF EXISTS ix_patristic_exegetical")

    # Extraction results indexes
    op.drop_index('ix_extraction_status', if_exists=True)
    op.drop_index('ix_extraction_agent_status', if_exists=True)
    op.drop_index('ix_extraction_type', if_exists=True)
    op.drop_index('ix_extraction_type_status', if_exists=True)
    op.drop_index('ix_extraction_confidence', if_exists=True)
    op.drop_index('ix_extraction_validated', if_exists=True)
    op.execute("DROP INDEX IF EXISTS ix_extraction_pending")
    op.execute("DROP INDEX IF EXISTS ix_extraction_failed")
