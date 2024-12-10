"""Rework vocabulary and keyword models

Revision ID: 757cb3c7680f
Revises: 75fd0f193787
Create Date: 2024-12-06 15:47:18.703326

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '757cb3c7680f'
down_revision = '75fd0f193787'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - adjusted ###
    op.create_table('keyword_audit',
                    sa.Column('id', sa.Integer(), sa.Identity(always=False), nullable=False),
                    sa.Column('client_id', sa.String(), nullable=False),
                    sa.Column('user_id', sa.String(), nullable=True),
                    sa.Column('command', postgresql.ENUM(name='auditcommand', create_type=False), nullable=False),
                    sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
                    sa.Column('_vocabulary_id', sa.String(), nullable=False),
                    sa.Column('_id', sa.Integer(), nullable=False),
                    sa.Column('_key', sa.String(), nullable=False),
                    sa.Column('_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
                    sa.Column('_status', sa.String(), nullable=False),
                    sa.Column('_parent_id', sa.Integer(), nullable=True),
                    sa.PrimaryKeyConstraint('id')
                    )
    op.create_table('keyword',
                    sa.Column('vocabulary_id', sa.String(), nullable=False),
                    sa.Column('id', sa.Integer(), sa.Identity(always=False, start=1001), nullable=False),
                    sa.Column('key', sa.String(), nullable=False),
                    sa.Column('data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
                    sa.Column('status', postgresql.ENUM(name='keywordstatus', create_type=False), nullable=False),
                    sa.Column('parent_id', sa.Integer(), nullable=True),
                    sa.ForeignKeyConstraint(['parent_id'], ['keyword.id'], ondelete='RESTRICT'),
                    sa.ForeignKeyConstraint(['vocabulary_id'], ['vocabulary.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('id'),
                    sa.UniqueConstraint('vocabulary_id', 'key')
                    )
    op.drop_table('vocabulary_term_audit')
    op.drop_table('vocabulary_term')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic ###
    op.create_table('vocabulary_term',
                    sa.Column('vocabulary_id', sa.VARCHAR(), autoincrement=False, nullable=False),
                    sa.Column('term_id', sa.VARCHAR(), autoincrement=False, nullable=False),
                    sa.Column('data', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=False),
                    sa.ForeignKeyConstraint(['vocabulary_id'], ['vocabulary.id'], name='vocabulary_term_vocabulary_id_fkey', ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('vocabulary_id', 'term_id', name='vocabulary_term_pkey')
                    )
    op.create_table('vocabulary_term_audit',
                    sa.Column('id', sa.INTEGER(),
                              sa.Identity(always=False, start=1, increment=1, minvalue=1, maxvalue=2147483647, cycle=False, cache=1),
                              autoincrement=True, nullable=False),
                    sa.Column('client_id', sa.VARCHAR(), autoincrement=False, nullable=False),
                    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=True),
                    sa.Column('command', postgresql.ENUM(name='auditcommand', create_type=False), nullable=False),
                    sa.Column('timestamp', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False),
                    sa.Column('_vocabulary_id', sa.VARCHAR(), autoincrement=False, nullable=False),
                    sa.Column('_term_id', sa.VARCHAR(), autoincrement=False, nullable=False),
                    sa.Column('_data', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=False),
                    sa.PrimaryKeyConstraint('id', name='vocabulary_term_audit_pkey')
                    )
    op.drop_table('keyword')
    op.drop_table('keyword_audit')
    # ### end Alembic commands ###
