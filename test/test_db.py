from random import randint

from sqlalchemy import select

from odp.const.db import ScopeType
from odp.db import Session
from odp.db.models import (Catalog, Collection, CollectionTag, Provider, Record, RecordTag,
                           Schema, Submission, Tag, Vocabulary, VocabularyTerm)
from test.factories import (CatalogFactory, CollectionFactory, CollectionTagFactory, ProviderFactory, RecordFactory,
                            RecordTagFactory, SchemaFactory, SubmissionFactory, TagFactory, VocabularyFactory)


def sorted_tuples(rows):
    return sorted(rows, key=lambda row: ''.join(str(val) for val in row))


def test_create_catalog():
    catalog = CatalogFactory()
    result = Session.execute(select(Catalog)).scalar_one()
    assert (result.id, result.url, result.data, result.timestamp) \
           == (catalog.id, catalog.url, catalog.data, catalog.timestamp)


def test_create_collection():
    collection = CollectionFactory()
    result = Session.execute(select(Collection, Provider).join(Provider)).one()
    assert (result.Collection.id, result.Collection.key, result.Collection.name, result.Collection.doi_key, result.Collection.provider_id, result.Provider.name) \
           == (collection.id, collection.key, collection.name, collection.doi_key, collection.provider.id, collection.provider.name)


def test_create_collection_tag():
    collection_tag = CollectionTagFactory()
    result = Session.execute(select(CollectionTag).join(Collection).join(Tag)).scalar_one()
    assert (result.collection_id, result.tag_id, result.tag_type, result.user_id, result.user_name, result.data) \
           == (collection_tag.collection.id, collection_tag.tag.id, 'collection', collection_tag.user_id, collection_tag.user_name, collection_tag.data)


def test_create_provider():
    provider = ProviderFactory()
    result = Session.execute(select(Provider)).scalar_one()
    assert (result.id, result.key, result.name) == (provider.id, provider.key, provider.name)


def test_create_record():
    record = RecordFactory(is_child_record=randint(0, 1))
    result = Session.execute(
        select(Record).where(Record.id == record.id)
    ).scalar_one()
    assert (
               result.id,
               result.doi,
               result.sid,
               result.metadata_,
               result.validity,
               result.collection_id,
               result.schema_id,
               result.schema_type,
               result.parent_id,
           ) == (
               record.id,
               record.doi,
               record.sid,
               record.metadata_,
               record.validity,
               record.collection.id,
               record.schema.id,
               record.schema.type,
               record.parent_id,
           )
    if record.parent_id:
        parent = Session.execute(
            select(Record).where(Record.id == record.parent_id)
        ).scalar_one()
        assert result.parent == parent
        assert result.parent_id == parent.id
        assert parent.children == [result]


def test_create_record_tag():
    record_tag = RecordTagFactory()
    result = Session.execute(select(RecordTag).join(Record).join(Tag)).scalar_one()
    assert (result.record_id, result.tag_id, result.tag_type, result.user_id, result.user_name, result.data) \
           == (record_tag.record.id, record_tag.tag.id, 'record', record_tag.user_id, record_tag.user_name, record_tag.data)


def test_create_schema():
    schema = SchemaFactory()
    result = Session.execute(
        select(Schema).
        where(Schema.id.notlike('vocab-schema-%'))  # ignore schemas created by vocabulary factories
    ).scalar_one()
    assert (result.id, result.type, result.uri) == (schema.id, schema.type, schema.uri)


def test_create_tag():
    tag = TagFactory()
    result = Session.execute(select(Tag)).scalar_one()
    assert (
               result.id,
               result.type,
               result.cardinality,
               result.public,
               result.schema_id,
               result.scope_id,
               result.vocabulary_id,
           ) == (
               tag.id,
               tag.type,
               tag.cardinality,
               tag.public,
               tag.schema_id,
               tag.scope_id,
               tag.vocabulary_id,
           )


def test_create_vocabulary():
    vocabulary = VocabularyFactory()
    result = Session.execute(select(Vocabulary, VocabularyTerm).join(VocabularyTerm))
    assert sorted_tuples((
                             row.Vocabulary.id,
                             row.Vocabulary.scope_id,
                             row.Vocabulary.schema_id,
                             row.Vocabulary.schema_type,
                             row.Vocabulary.static,
                             row.VocabularyTerm.vocabulary_id,
                             row.VocabularyTerm.term_id,
                             row.VocabularyTerm.data
                         ) for row in result) \
           == sorted_tuples((
                                vocabulary.id,
                                vocabulary.scope_id,
                                vocabulary.schema_id,
                                'vocabulary',
                                vocabulary.static,
                                term.vocabulary_id,
                                term.term_id,
                                term.data,
                            ) for term in vocabulary.terms)


def test_create_submission():
    submission = SubmissionFactory()
    result = Session.execute(
        select(Submission).where(Submission.id == submission.id)
    ).scalar_one()
    assert (
               result.id,
               result.doi,
               result.user_id,
               result.data,
               result.status,
               result.dataset_file_name,
               result.dataset_url,
               result.timestamp,
               result.collection_id,
               result.schema_id,
               result.record_id,
           ) == (
               submission.id,
               submission.doi,
               submission.user_id,
               submission.data,
               submission.status,
               submission.dataset_file_name,
               submission.dataset_url,
               submission.timestamp,
               submission.collection_id,
               submission.schema_id,
               submission.record_id,
           )
