"""decouple_auth_tables_for_keycloak

Revision ID: c4f91e7a2b10
Revises: 8a386c42418d
Create Date: 2026-10-20 14:25:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c4f91e7a2b10'
down_revision = '8a386c42418d'
branch_labels = None
depends_on = None


def upgrade():
    # 1. Add user_name columns to tag and audit tables (IF NOT EXISTS so it is safe
    #    even if non-destructive pre-migration SQL was already run)
    op.execute('ALTER TABLE collection_tag ADD COLUMN IF NOT EXISTS user_name VARCHAR')
    op.execute('ALTER TABLE record_tag ADD COLUMN IF NOT EXISTS user_name VARCHAR')

    op.execute('ALTER TABLE collection_audit ADD COLUMN IF NOT EXISTS user_name VARCHAR')
    op.execute('ALTER TABLE collection_tag_audit ADD COLUMN IF NOT EXISTS user_name VARCHAR')
    op.execute('ALTER TABLE collection_tag_audit ADD COLUMN IF NOT EXISTS _user_name VARCHAR')

    op.execute('ALTER TABLE record_audit ADD COLUMN IF NOT EXISTS user_name VARCHAR')
    op.execute('ALTER TABLE record_tag_audit ADD COLUMN IF NOT EXISTS user_name VARCHAR')
    op.execute('ALTER TABLE record_tag_audit ADD COLUMN IF NOT EXISTS _user_name VARCHAR')

    op.execute('ALTER TABLE provider_audit ADD COLUMN IF NOT EXISTS user_name VARCHAR')
    op.execute('ALTER TABLE vocabulary_term_audit ADD COLUMN IF NOT EXISTS user_name VARCHAR')

    # 2. Backfill user_name from public."user" before dropping the user table
    for table in (
        'collection_tag',
        'record_tag',
        'collection_audit',
        'collection_tag_audit',
        'record_audit',
        'record_tag_audit',
        'provider_audit',
        'vocabulary_term_audit',
    ):
        op.execute(f'''
            UPDATE {table} t
            SET user_name = u.name
            FROM "user" u
            WHERE t.user_id = u.id AND t.user_name IS NULL
        ''')

    for table in ('collection_tag_audit', 'record_tag_audit'):
        op.execute(f'''
            UPDATE {table} t
            SET _user_name = u.name
            FROM "user" u
            WHERE t._user_id = u.id AND t._user_name IS NULL
        ''')

    # 3. Drop foreign key constraints referencing "user" and "scope"
    op.execute('ALTER TABLE collection_tag DROP CONSTRAINT IF EXISTS collection_tag_user_id_fkey')
    op.execute('ALTER TABLE record_tag DROP CONSTRAINT IF EXISTS record_tag_user_id_fkey')
    op.execute('ALTER TABLE tag DROP CONSTRAINT IF EXISTS tag_scope_fkey')
    op.execute('ALTER TABLE vocabulary DROP CONSTRAINT IF EXISTS vocabulary_scope_fkey')

    # Also sever any legacy table FKs if present in the database
    op.execute('ALTER TABLE IF EXISTS package_tag DROP CONSTRAINT IF EXISTS package_tag_user_id_fkey')
    op.execute('ALTER TABLE IF EXISTS provider_user DROP CONSTRAINT IF EXISTS provider_user_user_id_fkey')
    op.execute('ALTER TABLE IF EXISTS archive DROP CONSTRAINT IF EXISTS archive_scope_fkey')

    # 4. Drop scope_type check constraints and columns from tag and vocabulary
    op.execute('ALTER TABLE tag DROP CONSTRAINT IF EXISTS tag_scope_type_check')
    op.execute('ALTER TABLE tag DROP COLUMN IF EXISTS scope_type')

    op.execute('ALTER TABLE vocabulary DROP CONSTRAINT IF EXISTS vocabulary_scope_type_check')
    op.execute('ALTER TABLE vocabulary DROP COLUMN IF EXISTS scope_type')

    # 5. Drop all 10 legacy auth tables
    for auth_table in (
        'identity_audit',
        'user_role',
        'role_scope',
        'role_collection',
        'client_scope',
        'client_collection',
        'user',
        'role',
        'client',
        'scope',
    ):
        op.execute(f'DROP TABLE IF EXISTS "{auth_table}" CASCADE')


def downgrade():
    raise NotImplementedError('Auth tables have been migrated to standalone Keycloak (saeon-auth).')

