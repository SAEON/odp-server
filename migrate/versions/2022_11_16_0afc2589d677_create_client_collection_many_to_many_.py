"""Create client-collection many-to-many association

Revision ID: 0afc2589d677
Revises: 
Create Date: 2022-11-16 08:20:49.354131

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0afc2589d677'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic ###
    op.create_table('client_collection',
    sa.Column('client_id', sa.String(), nullable=False),
    sa.Column('collection_id', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['client_id'], ['client.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['collection_id'], ['collection.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('client_id', 'collection_id')
    )
    op.add_column('client', sa.Column('collection_specific', sa.Boolean(), server_default='false', nullable=False))
    op.drop_constraint('client_collection_id_fkey', 'client', type_='foreignkey')
    op.drop_column('client', 'collection_id')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic ###
    op.add_column('client', sa.Column('collection_id', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.create_foreign_key('client_collection_id_fkey', 'client', 'collection', ['collection_id'], ['id'], onupdate='CASCADE', ondelete='CASCADE')
    op.drop_column('client', 'collection_specific')
    op.drop_table('client_collection')
    # ### end Alembic commands ###
