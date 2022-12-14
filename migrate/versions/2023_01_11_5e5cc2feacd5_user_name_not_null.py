"""User name not null

Revision ID: 5e5cc2feacd5
Revises: f1913131f924
Create Date: 2023-01-11 10:00:40.398696

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5e5cc2feacd5'
down_revision = 'f1913131f924'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic ###
    op.alter_column('user', 'name',
               existing_type=sa.VARCHAR(),
               nullable=False)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic ###
    op.alter_column('user', 'name',
               existing_type=sa.VARCHAR(),
               nullable=True)
    # ### end Alembic commands ###
