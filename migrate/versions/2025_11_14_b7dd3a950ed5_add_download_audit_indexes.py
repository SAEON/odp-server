"""add_download_audit_indexes

Revision ID: b7dd3a950ed5
Revises: 303267e1f361
Create Date: 2025-11-14 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'b7dd3a950ed5'
down_revision = '303267e1f361'
branch_labels = None
depends_on = None


def upgrade():
    # Index for date-range queries (most common reporting use case)
    op.create_index(
        'idx_download_audit_timestamp',
        'download_audit',
        [sa.desc('timestamp')],
        postgresql_using='btree'
    )

    # Index for client/user tracking
    op.create_index(
        'idx_download_audit_client_id',
        'download_audit',
        ['client_id'],
        postgresql_using='btree'
    )

    # GIN indexes on meta JSON fields for flexible searching
    # Email search in meta
    op.create_index(
        'idx_download_audit_meta_email',
        'download_audit',
        [sa.text("(meta->'email')")],
        postgresql_using='gin'
    )

    # Organisation search in meta
    op.create_index(
        'idx_download_audit_meta_organisation',
        'download_audit',
        [sa.text("(meta->'organisation')")],
        postgresql_using='gin'
    )

    # Download type search in meta
    op.create_index(
        'idx_download_audit_meta_download_type',
        'download_audit',
        [sa.text("(meta->'download_type')")],
        postgresql_using='gin'
    )

    # Full meta object index for flexible querying
    op.create_index(
        'idx_download_audit_meta',
        'download_audit',
        ['meta'],
        postgresql_using='gin'
    )

    # Composite index for common filter combinations
    # (timestamp DESC, client_id) for user activity over time
    op.create_index(
        'idx_download_audit_timestamp_client_id',
        'download_audit',
        [sa.text('timestamp DESC'), 'client_id'],
        postgresql_using='btree'
    )


def downgrade():
    op.drop_index('idx_download_audit_timestamp_client_id')
    op.drop_index('idx_download_audit_meta')
    op.drop_index('idx_download_audit_meta_download_type')
    op.drop_index('idx_download_audit_meta_organisation')
    op.drop_index('idx_download_audit_meta_email')
    op.drop_index('idx_download_audit_client_id')
    op.drop_index('idx_download_audit_timestamp')
