"""Create catalog-collection relation

Revision ID: 213da8b3a737
Revises: 503fc5a0b3f2
Create Date: 2022-12-12 07:45:17.130947

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '213da8b3a737'
down_revision = '503fc5a0b3f2'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic ###
    op.create_table('catalog_collection',
    sa.Column('catalog_id', sa.String(), nullable=False),
    sa.Column('collection_id', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['catalog_id'], ['catalog.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['collection_id'], ['collection.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('catalog_id', 'collection_id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic ###
    op.drop_table('catalog_collection')
    # ### end Alembic commands ###