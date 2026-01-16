"""Initial schema for BIBLOS v2

Revision ID: 001
Revises:
Create Date: 2026-01-15

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Books table
    op.create_table(
        'books',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('code', sa.String(3), nullable=False),  # GEN, EXO, MAT, etc.
        sa.Column('name', sa.String(50), nullable=False),
        sa.Column('testament', sa.String(3), nullable=False),  # OT, NT
        sa.Column('order_num', sa.Integer(), nullable=False),
        sa.Column('chapter_count', sa.Integer(), nullable=False),
        sa.Column('verse_count', sa.Integer(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code')
    )
    op.create_index('ix_books_code', 'books', ['code'])

    # Verses table
    op.create_table(
        'verses',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('reference', sa.String(20), nullable=False),  # GEN.1.1
        sa.Column('book_id', sa.Integer(), nullable=False),
        sa.Column('chapter', sa.Integer(), nullable=False),
        sa.Column('verse', sa.Integer(), nullable=False),
        sa.Column('text_greek', sa.Text(), nullable=True),
        sa.Column('text_hebrew', sa.Text(), nullable=True),
        sa.Column('text_english', sa.Text(), nullable=True),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('morphology', postgresql.JSONB(), nullable=True),
        sa.Column('syntax', postgresql.JSONB(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['book_id'], ['books.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('reference')
    )
    op.create_index('ix_verses_reference', 'verses', ['reference'])
    op.create_index('ix_verses_book_chapter', 'verses', ['book_id', 'chapter'])

    # Cross-references table
    op.create_table(
        'cross_references',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_id', sa.Integer(), nullable=False),
        sa.Column('target_id', sa.Integer(), nullable=False),
        sa.Column('connection_type', sa.String(20), nullable=False),
        sa.Column('strength', sa.String(10), nullable=False),  # strong, moderate, weak
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('bidirectional', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('patristic_support', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('keywords', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['source_id'], ['verses.id']),
        sa.ForeignKeyConstraint(['target_id'], ['verses.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_crossref_source', 'cross_references', ['source_id'])
    op.create_index('ix_crossref_target', 'cross_references', ['target_id'])
    op.create_index('ix_crossref_type', 'cross_references', ['connection_type'])
    op.create_unique_constraint(
        'uq_crossref_pair',
        'cross_references',
        ['source_id', 'target_id', 'connection_type']
    )

    # Patristic citations table
    op.create_table(
        'patristic_citations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('verse_id', sa.Integer(), nullable=False),
        sa.Column('father_name', sa.String(100), nullable=False),
        sa.Column('work_title', sa.String(200), nullable=False),
        sa.Column('citation_ref', sa.String(100), nullable=True),
        sa.Column('text_greek', sa.Text(), nullable=True),
        sa.Column('text_english', sa.Text(), nullable=True),
        sa.Column('category', sa.String(50), nullable=True),  # exegetical, homiletic, doctrinal
        sa.Column('century', sa.Integer(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['verse_id'], ['verses.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_patristic_verse', 'patristic_citations', ['verse_id'])
    op.create_index('ix_patristic_father', 'patristic_citations', ['father_name'])

    # Extraction results table
    op.create_table(
        'extraction_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('verse_id', sa.Integer(), nullable=False),
        sa.Column('agent_name', sa.String(50), nullable=False),
        sa.Column('extraction_type', sa.String(30), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('data', postgresql.JSONB(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['verse_id'], ['verses.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_extraction_verse_agent', 'extraction_results', ['verse_id', 'agent_name'])


def downgrade() -> None:
    op.drop_table('extraction_results')
    op.drop_table('patristic_citations')
    op.drop_table('cross_references')
    op.drop_table('verses')
    op.drop_table('books')
    op.execute('DROP EXTENSION IF EXISTS vector')
