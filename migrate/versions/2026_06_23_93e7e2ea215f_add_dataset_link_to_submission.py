"""add_dataset_link_to_submission

Revision ID: 93e7e2ea215f
Revises: 8a386c42418d
Create Date: 2026-06-23 10:05:49.040363

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '93e7e2ea215f'
down_revision = '8a386c42418d'
branch_labels = None
depends_on = None


def upgrade():
    # Add the dataset_url column to the submission table
    op.add_column('submission', sa.Column('dataset_url', sa.String(), nullable=True))


def downgrade():
    # Remove the dataset_url column if rolling back
    op.drop_column('submission', 'dataset_url')