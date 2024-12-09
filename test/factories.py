import re
import sys
from datetime import datetime, timezone
from random import choice, randint

import factory
from factory.alchemy import SQLAlchemyModelFactory
from faker import Faker
from sqlalchemy.orm import scoped_session, sessionmaker

import odp.db
from odp.db.models import (
    Archive,
    ArchiveResource,
    Catalog,
    Client,
    Collection,
    CollectionTag,
    Keyword,
    Package,
    PackageTag,
    Provider,
    Record,
    RecordTag,
    Resource,
    Role,
    Schema,
    Scope,
    Tag,
    User,
    Vocabulary,
    VocabularyTerm,
)
from test import datacite4_example, iso19115_example

FactorySession = scoped_session(sessionmaker(
    bind=odp.db.engine,
    autocommit=False,
    autoflush=False,
    future=True,
))

fake = Faker()


def id_from_name(obj):
    name, _, n = obj.name.rpartition('.')
    prefix, _, _ = name.partition(' ')
    return f'{_sanitize_id(prefix)}.{n}'


def id_from_fake(src, n):
    fake_val = getattr(fake, src)()
    return f'{_sanitize_id(fake_val)}.{n}'


def _sanitize_id(val):
    return re.sub(r'[^-.:\w]', '_', val)


def create_metadata(record, n):
    if record.use_example_metadata:
        if record.schema_id == 'SAEON.DataCite4':
            metadata = datacite4_example()
        elif record.schema_id == 'SAEON.ISO19115':
            metadata = iso19115_example()
    else:
        metadata = {'foo': f'test-{n}'}

    if record.doi:
        metadata |= {'doi': record.doi}
    else:
        metadata.pop('doi', None)

    if record.parent_doi:
        metadata.setdefault("relatedIdentifiers", [])
        metadata["relatedIdentifiers"] += [{
            "relatedIdentifier": record.parent_doi,
            "relatedIdentifierType": "DOI",
            "relationType": "IsPartOf"
        }]

    # non-DOI relatedIdentifierType should be ignored for parent_id calculation
    if not record.use_example_metadata and randint(0, 1):
        metadata.setdefault("relatedIdentifiers", [])
        metadata["relatedIdentifiers"] += [{
            "relatedIdentifier": "foo",
            "relatedIdentifierType": "URL",
            "relationType": "IsPartOf"
        }]

    # non-IsPartOf relationType should be ignored for parent_id calculation
    if not record.use_example_metadata and randint(0, 1):
        metadata.setdefault("relatedIdentifiers", [])
        metadata["relatedIdentifiers"] += [{
            "relatedIdentifier": "bar",
            "relatedIdentifierType": "DOI",
            "relationType": "HasPart"
        }]

    return metadata


def schema_uri_from_type(schema):
    if schema.type == 'metadata':
        return choice((
            'https://odp.saeon.ac.za/schema/metadata/saeon/datacite4',
            'https://odp.saeon.ac.za/schema/metadata/saeon/iso19115',
            'https://odp.saeon.ac.za/schema/metadata/datacite/kernel-4.3',
        ))
    elif schema.type == 'tag':
        return choice((
            'https://odp.saeon.ac.za/schema/tag/generic',
            'https://odp.saeon.ac.za/schema/tag/record/migrated',
            'https://odp.saeon.ac.za/schema/tag/record/qc',
            'https://odp.saeon.ac.za/schema/tag/record/embargo',
            'https://odp.saeon.ac.za/schema/tag/collection/infrastructure',
            'https://odp.saeon.ac.za/schema/tag/collection/project',
        ))
    elif schema.type == 'keyword':
        return choice((
            'https://odp.saeon.ac.za/schema/keyword/institution',
        ))
    elif schema.type == 'vocabulary':
        return choice((
            'https://odp.saeon.ac.za/schema/vocabulary/infrastructure',
            'https://odp.saeon.ac.za/schema/vocabulary/project',
        ))
    else:
        return fake.uri()


class ODPModelFactory(SQLAlchemyModelFactory):
    class Meta:
        sqlalchemy_session = FactorySession
        sqlalchemy_session_persistence = 'commit'


class ScopeFactory(ODPModelFactory):
    class Meta:
        model = Scope
        sqlalchemy_get_or_create = ('id', 'type')  # Force the factory to fetch if keys match

    id = factory.Sequence(lambda n: f'{fake.word()}.{n}')
    type = factory.LazyFunction(lambda: choice(('odp', 'oauth', 'client')))


class SchemaFactory(ODPModelFactory):
    class Meta:
        model = Schema

    id = factory.Sequence(lambda n: f'{fake.word()}.{n}')
    type = factory.LazyFunction(lambda: choice(('metadata', 'tag', 'keyword', 'vocabulary')))
    uri = factory.LazyAttribute(schema_uri_from_type)
    md5 = factory.Faker('md5')
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))

    @factory.post_generation
    def create_vocabulary_for_tag_schema(obj, create, extracted):
        """Create vocabulary objects as needed for tag schemas, so that
        ``vocabulary`` keyword references work."""
        if obj.type == 'tag':
            for vocab_id in 'Infrastructure', 'Project':
                if obj.uri.endswith(vocab_id.lower()) and not FactorySession.get(Vocabulary, vocab_id):
                    VocabularyFactory(
                        id=vocab_id,
                        schema=SchemaFactory(
                            id=factory.Sequence(lambda n: f'vocab-schema-{fake.word()}.{n}'),
                            type='vocabulary',
                        )
                    )


class CatalogFactory(ODPModelFactory):
    class Meta:
        model = Catalog

    id = factory.Sequence(lambda n: f'{fake.slug()}.{n}')
    url = factory.Faker('url')
    data = factory.Sequence(lambda n: dict(foo=f'{fake.catch_phrase()}.{n}'))
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))


class ProviderFactory(ODPModelFactory):
    class Meta:
        model = Provider

    id = factory.Faker('uuid4')
    key = factory.LazyAttribute(id_from_name)
    name = factory.Sequence(lambda n: f'{fake.company()}.{n}')
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))

    @factory.post_generation
    def users(obj, create, users):
        if users:
            for user in users:
                obj.users.append(user)
            if create:
                FactorySession.commit()


class PackageFactory(ODPModelFactory):
    class Meta:
        model = Package

    id = factory.Faker('uuid4')
    key = factory.LazyAttribute(lambda p: re.sub(r'\W', '_', p.title))
    title = factory.Sequence(lambda n: f'{fake.catch_phrase()}.{n}')
    status = factory.LazyFunction(lambda: choice(('pending', 'submitted', 'archived', 'deleted')))
    provider = factory.SubFactory(ProviderFactory)
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))


class ResourceFactory(ODPModelFactory):
    class Meta:
        model = Resource

    id = factory.Faker('uuid4')
    folder = factory.LazyFunction(lambda: choice(('', f'{fake.uri_path(deep=randint(1, 4))}')))
    filename = factory.Sequence(lambda n: f'{fake.file_name()}.{n}')
    title = factory.Faker('catch_phrase')
    description = factory.Faker('sentence')
    mimetype = factory.Faker('mime_type')
    size = factory.LazyFunction(lambda: randint(1, sys.maxsize))
    hash = factory.LazyAttribute(lambda r: fake.md5() if r.hash_algorithm == 'md5' else fake.sha256())
    hash_algorithm = factory.LazyFunction(lambda: choice(('md5', 'sha256')))
    package = factory.SubFactory(PackageFactory)
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))


class CollectionFactory(ODPModelFactory):
    class Meta:
        model = Collection

    id = factory.Faker('uuid4')
    key = factory.LazyAttribute(id_from_name)
    name = factory.Sequence(lambda n: f'{fake.catch_phrase()}.{n}')
    doi_key = factory.LazyFunction(lambda: fake.word() if randint(0, 1) else None)
    provider = factory.SubFactory(ProviderFactory)
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))


class ClientFactory(ODPModelFactory):
    class Meta:
        model = Client

    id = factory.Sequence(lambda n: id_from_fake('catch_phrase', n))
    provider_specific = factory.LazyFunction(lambda: randint(0, 1))
    provider = factory.Maybe(
        'provider_specific',
        yes_declaration=factory.SubFactory(ProviderFactory),
        no_declaration=None,
    )

    @factory.post_generation
    def scopes(obj, create, scopes):
        if scopes:
            for scope in scopes:
                obj.scopes.append(scope)
            if create:
                FactorySession.commit()


class KeywordFactory(ODPModelFactory):
    class Meta:
        model = Keyword
        exclude = ('has_child_schema',)

    parent_id = None
    id = factory.LazyAttributeSequence(lambda k, n: (f'{k.parent_id}/' if k.parent_id else '') + f'{fake.word()}.{n}')
    data = factory.LazyFunction(lambda: {fake.word(): fake.word()})
    status = factory.LazyFunction(lambda: choice(('proposed', 'approved', 'rejected')))

    has_child_schema = factory.LazyAttribute(lambda k: '/' not in k.id or randint(0, 1))
    child_schema = factory.Maybe(
        'has_child_schema',
        yes_declaration=factory.SubFactory(SchemaFactory, type='keyword'),
        no_declaration=None,
    )
    child_schema_id = factory.LazyAttribute(lambda k: k.child_schema.id if k.child_schema else None)
    child_schema_type = factory.LazyAttribute(lambda k: 'keyword' if k.child_schema else None)

    @factory.post_generation
    def children(obj, create, _):
        if len(obj.id.split('/')) < 4:
            add_child = KeywordFactory.create if create else KeywordFactory.build
            for n in range(randint(0, 5)):
                add_child(parent_id=obj.id)


class VocabularyTermFactory(ODPModelFactory):
    class Meta:
        model = VocabularyTerm

    vocabulary = None
    term_id = factory.Sequence(lambda n: id_from_fake('word', n))
    data = factory.LazyAttribute(lambda t: {'id': t.term_id})


class VocabularyFactory(ODPModelFactory):
    class Meta:
        model = Vocabulary

    id = factory.Sequence(lambda n: id_from_fake('word', n))
    scope = factory.SubFactory(ScopeFactory, type='odp')
    schema = factory.SubFactory(SchemaFactory, type='vocabulary')
    static = factory.LazyFunction(lambda: randint(0, 1))
    terms = factory.RelatedFactoryList(
        VocabularyTermFactory,
        factory_related_name='vocabulary',
        size=lambda: randint(4, 8),
    )


class TagFactory(ODPModelFactory):
    class Meta:
        model = Tag
        exclude = ('is_keyword_tag',)

    id = factory.LazyAttribute(lambda tag: f'tag-{tag.scope.id}')
    type = factory.LazyFunction(lambda: choice(('collection', 'record')))
    cardinality = factory.LazyFunction(lambda: choice(('one', 'user', 'multi')))
    public = factory.LazyFunction(lambda: randint(0, 1))
    scope = factory.SubFactory(ScopeFactory, type='odp')
    schema = factory.SubFactory(SchemaFactory, type='tag')

    is_keyword_tag = factory.LazyFunction(lambda: randint(0, 1))
    vocabulary = factory.Maybe(
        'is_keyword_tag',
        yes_declaration=factory.SubFactory(VocabularyFactory),
        no_declaration=None,
    )


class UserFactory(ODPModelFactory):
    class Meta:
        model = User

    id = factory.Faker('uuid4')
    name = factory.Faker('name')
    email = factory.Sequence(lambda n: f'{fake.email()}.{n}')
    active = factory.LazyFunction(lambda: randint(0, 1))
    verified = factory.LazyFunction(lambda: randint(0, 1))
    picture = factory.Faker('image_url')

    @factory.post_generation
    def roles(obj, create, roles):
        if roles:
            for role in roles:
                obj.roles.append(role)
            if create:
                FactorySession.commit()


class CollectionTagFactory(ODPModelFactory):
    class Meta:
        model = CollectionTag

    collection = factory.SubFactory(CollectionFactory)
    tag = factory.SubFactory(TagFactory, type='collection')
    user = factory.SubFactory(UserFactory)
    data = {}
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))


class PackageTagFactory(ODPModelFactory):
    class Meta:
        model = PackageTag

    package = factory.SubFactory(PackageFactory)
    tag = factory.SubFactory(TagFactory, type='package')
    user = factory.SubFactory(UserFactory)
    data = factory.LazyFunction(lambda: {'foo': fake.word()})
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))


class RecordFactory(ODPModelFactory):
    class Meta:
        model = Record
        exclude = ('identifiers', 'is_child_record', 'parent_doi', 'use_example_metadata')

    identifiers = factory.LazyFunction(lambda: choice(('doi', 'sid', 'both')))
    doi = factory.LazyAttributeSequence(lambda r, n: f'10.5555/Test-{n}' if r.identifiers in ('doi', 'both') else None)
    sid = factory.LazyAttributeSequence(lambda r, n: f'test-{n}' if r.doi is None or r.identifiers in ('sid', 'both') else None)

    parent_doi = None
    use_example_metadata = False
    metadata_ = factory.LazyAttributeSequence(create_metadata)
    validity = factory.LazyAttribute(lambda r: dict(valid=r.use_example_metadata))

    collection = factory.SubFactory(CollectionFactory)
    schema_id = factory.LazyFunction(lambda: choice(('SAEON.DataCite4', 'SAEON.ISO19115')))
    schema_type = 'metadata'
    schema = factory.LazyAttribute(lambda r: FactorySession.get(Schema, (r.schema_id, 'metadata')) or
                                             SchemaFactory(id=r.schema_id, type='metadata'))
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))

    is_child_record = False
    parent = factory.Maybe(
        'is_child_record',
        yes_declaration=factory.SubFactory('test.factories.RecordFactory'),
        no_declaration=None,
    )


class RecordTagFactory(ODPModelFactory):
    class Meta:
        model = RecordTag

    record = factory.SubFactory(RecordFactory)
    tag = factory.SubFactory(TagFactory, type='record')
    user = factory.SubFactory(UserFactory)
    data = {}
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))


class RoleFactory(ODPModelFactory):
    class Meta:
        model = Role

    id = factory.Sequence(lambda n: id_from_fake('job', n))
    collection_specific = factory.LazyFunction(lambda: randint(0, 1))

    @factory.post_generation
    def scopes(obj, create, scopes):
        if scopes:
            for scope in scopes:
                obj.scopes.append(scope)
            if create:
                FactorySession.commit()

    @factory.post_generation
    def collections(obj, create, collections):
        if collections:
            for collection in collections:
                obj.collections.append(collection)
            if create:
                FactorySession.commit()


class ArchiveFactory(ODPModelFactory):
    class Meta:
        model = Archive

    id = factory.Sequence(lambda n: f'{fake.slug()}.{n}')
    download_url = factory.Faker('url')
    upload_url = factory.Faker('url')
    adapter = factory.LazyFunction(lambda: choice(('filesystem', 'nextcloud', 'website')))
    scope = factory.SubFactory(ScopeFactory, type='odp')


class ArchiveResourceFactory(ODPModelFactory):
    class Meta:
        model = ArchiveResource

    archive = factory.SubFactory(ArchiveFactory)
    resource = factory.SubFactory(ResourceFactory)
    path = factory.LazyAttribute(lambda a: f'{fake.uri_path(deep=randint(1, 5))}/{a.resource.filename}')
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
