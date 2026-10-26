import logging
import os
import pathlib
from datetime import datetime, timezone

import yaml
from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
from sqlalchemy import delete, select, text
from sqlalchemy.exc import ProgrammingError

from odp.const import (
    ODPCatalog,
    ODPCollectionTag,
    ODPMetadataSchema,
    ODPRecordTag,
    ODPTagSchema,
    ODPVocabulary,
    ODPVocabularySchema,
)
from odp.const.db import SchemaType
from odp.db import Base, Session, engine
from odp.db.models import Catalog, Schema, Tag, Vocabulary, VocabularyTerm
from odp.lib.schema import schema_md5

datadir = pathlib.Path(__file__).parent / 'systemdata'
logger = logging.getLogger(__name__)


def initialize():
    logger.info('Initializing static system data...')

    load_dotenv(pathlib.Path(os.getcwd()) / '.env')  # for a local run; in a container there's no .env

    init_database_schema()

    with Session.begin():
        init_schemas()
        init_vocabularies()
        init_tags()
        init_catalogs()

    logger.info('Done.')


def init_database_schema():
    """Create or update the ODP database schema."""
    cwd = os.getcwd()
    os.chdir(pathlib.Path(__file__).parent)
    try:
        alembic_cfg = Config('alembic.ini')
        try:
            with engine.connect() as conn:
                conn.execute(text('select version_num from alembic_version'))
            schema_exists = True
        except ProgrammingError:  # from psycopg2.errors.UndefinedTable
            schema_exists = False

        if schema_exists:
            command.upgrade(alembic_cfg, 'head')
        else:
            Base.metadata.create_all(engine)
            command.stamp(alembic_cfg, 'head')
            logger.info('Created the ODP database schema.')
    finally:
        os.chdir(cwd)


def init_schemas():
    """Create or update schema definitions."""
    with open(datadir / 'schemas.yml') as f:
        schema_data = yaml.safe_load(f)

    for schema_id in (schema_ids := [s.value for s in ODPMetadataSchema] +
                                    [s.value for s in ODPTagSchema] +
                                    [s.value for s in ODPVocabularySchema]):
        schema_spec = schema_data[schema_id]
        schema_type = schema_spec['type']
        schema = Session.get(Schema, (schema_id, schema_type)) or Schema(id=schema_id, type=schema_type)
        schema.uri = schema_spec['uri']

        if (md5 := schema_md5(schema.uri)) != schema.md5:
            schema.md5 = md5
            schema.timestamp = datetime.now(timezone.utc)
            logger.info(f'Updated MD5 and timestamp for schema {schema_id}')

        schema.save()

    if orphaned_yml_schemas := [schema_id for schema_id in schema_data if schema_id not in schema_ids]:
        logger.warning(f'Orphaned schema definitions in schemas.yml {orphaned_yml_schemas}')

    if orphaned_db_schemas := Session.execute(select(Schema.id).where(Schema.id.not_in(schema_ids))).scalars().all():
        logger.warning(f'Orphaned schema definitions in schema table {orphaned_db_schemas}')


def init_tags():
    """Create or update tag definitions."""
    with open(datadir / 'tags.yml') as f:
        tag_data = yaml.safe_load(f)

    for tag_id in (tag_ids := [t.value for t in ODPRecordTag] + [t.value for t in ODPCollectionTag]):
        tag_spec = tag_data[tag_id]
        tag_type = tag_spec['type']
        tag = Session.get(Tag, (tag_id, tag_type)) or Tag(id=tag_id, type=tag_type)
        tag.cardinality = tag_spec['cardinality']
        tag.public = tag_spec['public']
        tag.scope_id = tag_spec['scope_id']
        tag.schema_id = tag_spec['schema_id']
        tag.schema_type = SchemaType.tag
        tag.vocabulary_id = tag_spec.get('vocabulary_id')
        tag.save()

    if orphaned_yml_tags := [tag_id for tag_id in tag_data if tag_id not in tag_ids]:
        logger.warning(f'Orphaned tag definitions in tags.yml {orphaned_yml_tags}')

    if orphaned_db_tags := Session.execute(select(Tag.id).where(Tag.id.not_in(tag_ids))).scalars().all():
        logger.warning(f'Orphaned tag definitions in tag table {orphaned_db_tags}')


def init_vocabularies():
    """Create or update vocabulary definitions.

    If `static_terms` are declared in the .yml, the vocabulary is flagged
    as static, and its terms are maintained here without audit logging.
    """
    with open(datadir / 'vocabularies.yml') as f:
        vocabulary_data = yaml.safe_load(f)

    for vocabulary_id in (vocabulary_ids := [v.value for v in ODPVocabulary]):
        vocabulary_spec = vocabulary_data[vocabulary_id]
        vocabulary = Session.get(Vocabulary, vocabulary_id) or Vocabulary(id=vocabulary_id)
        vocabulary.scope_id = vocabulary_spec['scope_id']
        vocabulary.schema_id = vocabulary_spec['schema_id']
        vocabulary.schema_type = SchemaType.vocabulary
        vocabulary.save()

        if static_terms := vocabulary_spec.get('static_terms'):
            vocabulary.static = True
            term_ids = []
            for term_spec in static_terms:
                term_ids += [term_id := term_spec['id']]
                term = Session.get(VocabularyTerm, (vocabulary_id, term_id)) or VocabularyTerm(
                    vocabulary_id=vocabulary_id,
                    term_id=term_id,
                )
                term.data = term_spec
                term.save()

                Session.execute(
                    delete(VocabularyTerm).
                    where(VocabularyTerm.vocabulary_id == vocabulary_id).
                    where(VocabularyTerm.term_id.not_in(term_ids))
                )
        else:
            vocabulary.static = False

    if orphaned_yml_vocabularies := [vocabulary_id for vocabulary_id in vocabulary_data if vocabulary_id not in vocabulary_ids]:
        logger.warning(f'Orphaned vocabulary definitions in vocabularies.yml {orphaned_yml_vocabularies}')

    if orphaned_db_vocabularies := Session.execute(select(Vocabulary.id).where(Vocabulary.id.not_in(vocabulary_ids))).scalars().all():
        logger.warning(f'Orphaned vocabulary definitions in vocabulary table {orphaned_db_vocabularies}')


def init_catalogs():
    """Create or update catalog definitions."""
    with open(datadir / 'catalogs.yml') as f:
        catalog_data = yaml.safe_load(f)

    for catalog_id in (catalog_ids := [c.value for c in ODPCatalog]):
        catalog_spec = catalog_data[catalog_id]
        catalog = Session.get(Catalog, catalog_id) or Catalog(id=catalog_id)
        catalog.url = os.environ[catalog_spec['url_env']]
        catalog.save()

    if orphaned_yml_catalogs := [catalog_id for catalog_id in catalog_data if catalog_id not in catalog_ids]:
        logger.warning(f'Orphaned catalog definitions in catalogs.yml {orphaned_yml_catalogs}')

    if orphaned_db_catalogs := Session.execute(select(Catalog.id).where(Catalog.id.not_in(catalog_ids))).scalars().all():
        logger.warning(f'Orphaned catalog definitions in catalog table {orphaned_db_catalogs}')
