"""Modify package status

Revision ID: 326d3fc9b55b
Revises: c8363345ee8c
Create Date: 2025-05-06 11:16:12.242684

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '326d3fc9b55b'
down_revision = 'c8363345ee8c'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_column('package', 'status')
    op.execute('drop type packagestatus')
    op.execute("create type packagestatus as enum ('editing', 'submitted', 'in_review', 'archived', 'delete_pending')")
    op.add_column('package', sa.Column('status', postgresql.ENUM(name='packagestatus', create_type=False), nullable=False))


def downgrade():
    op.drop_column('package', 'status')
    op.execute('drop type packagestatus')
    op.execute("create type packagestatus as enum ('pending', 'submitted', 'archived', 'deleted')")
    op.add_column('package', sa.Column('status', postgresql.ENUM(name='packagestatus', create_type=False), nullable=False))
