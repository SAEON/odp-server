"""Split resource status

Revision ID: b9774b38ac0b
Revises: 660293c24387
Create Date: 2025-04-16 17:49:32.137456

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = 'b9774b38ac0b'
down_revision = '660293c24387'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_column('archive_resource', 'status')
    op.execute('drop type resourcestatus')
    op.execute("create type archiveresourcestatus as enum ('pending', 'valid', 'missing', 'corrupt')")
    op.add_column('archive_resource', sa.Column('status', postgresql.ENUM(name='archiveresourcestatus', create_type=False), nullable=False))
    op.execute("create type resourcestatus as enum ('active', 'delete_pending')")
    op.add_column('resource', sa.Column('status', postgresql.ENUM(name='resourcestatus', create_type=False), nullable=False))


def downgrade():
    op.drop_column('resource', 'status')
    op.execute('drop type resourcestatus')
    op.drop_column('archive_resource', 'status')
    op.execute('drop type archiveresourcestatus')
    op.execute("create type resourcestatus as enum ('pending', 'valid', 'missing', 'corrupt', 'delete_pending')")
    op.add_column('archive_resource', sa.Column('status', postgresql.ENUM(name='resourcestatus', create_type=False), nullable=False))
