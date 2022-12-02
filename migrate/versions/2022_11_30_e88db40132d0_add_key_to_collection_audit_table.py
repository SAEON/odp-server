"""Add key to collection audit table

Revision ID: e88db40132d0
Revises: da9bb40789b0
Create Date: 2022-11-30 16:06:55.276123

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e88db40132d0'
down_revision = 'da9bb40789b0'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic ###
    op.add_column('collection_audit', sa.Column('_key', sa.String(), nullable=False))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic ###
    op.drop_column('collection_audit', '_key')
    # ### end Alembic commands ###