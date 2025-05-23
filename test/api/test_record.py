import uuid
from datetime import datetime
from random import randint

import pytest
from sqlalchemy import select

from odp.const import ODPCollectionTag, ODPScope
from odp.db.models import CollectionTag, PublishedRecord, Record, RecordAudit, RecordTag, User
from test import TestSession
from test.api import all_scopes, all_scopes_excluding
from test.api.assertions import assert_conflict, assert_forbidden, assert_new_timestamp, assert_not_found, assert_ok_null, assert_unprocessable
from test.api.assertions.tags import (
    assert_tag_instance_audit_log,
    assert_tag_instance_audit_log_empty,
    assert_tag_instance_db_state,
    assert_tag_instance_output,
    keyword_tag_args,
    new_generic_tag,
)
from test.api.conftest import try_skip_collection_constraint
from test.factories import (
    CollectionFactory,
    CollectionTagFactory,
    FactorySession,
    RecordFactory,
    RecordTagFactory,
    TagFactory,
)


@pytest.fixture
def record_batch():
    """Create and commit a batch of Record instances."""
    records = []
    for n in range(randint(3, 5)):
        kwargs = {}
        if n == 0:
            # ensure record 0 has a DOI; it can be used as a parent record
            kwargs |= dict(identifiers='doi')
        elif randint(0, 1):
            # optionally make this a child of record 0; the child must also have a DOI
            kwargs |= dict(identifiers='doi', parent=records[0])

        records += [record := RecordFactory(**kwargs)]
        RecordTagFactory.create_batch(randint(0, 3), record=record)
        CollectionTagFactory.create_batch(randint(0, 3), collection=record.collection)
    return records


@pytest.fixture
def record_batch_no_tags():
    """Create and commit a batch of Record instances
    without any tag instances."""
    return [RecordFactory() for _ in range(randint(3, 5))]


@pytest.fixture
def record_batch_with_ids():
    """Create and commit a batch of Record instances
    with both DOIs and SIDs."""
    return [RecordFactory(identifiers='both') for _ in range(randint(3, 5))]


def record_build(collection=None, collection_tags=None, **id):
    """Build and return an uncommitted Record instance.
    Referenced collection and any collection tags are however committed."""
    record = RecordFactory.build(
        **id,
        collection=collection or (collection := CollectionFactory()),
        collection_id=collection.id,
    )
    if collection_tags:
        for ct in collection_tags:
            CollectionTagFactory(
                collection=record.collection,
                tag=TagFactory(id=ct, type='collection'),
            )
    return record


@pytest.fixture(params=[False, True])
def is_admin_route(request):
    return request.param


@pytest.fixture(params=['doi', 'sid', 'both'])
def ident_conflict(request):
    return request.param


@pytest.fixture(params=['change', 'remove'])
def doi_change(request):
    return request.param


@pytest.fixture(params=[None, 'id', 'doi'])
def is_published_record(request):
    return request.param


@pytest.fixture(params=[True, False])
def is_same_user(request):
    return request.param


@pytest.fixture(params=[None, 'doi', 'doi.org'])
def with_parent(request):
    return request.param


@pytest.fixture(params=['create', 'update'])
def create_or_update(request):
    return request.param


@pytest.fixture(params=['non-doi', 'multiple-parents', 'parent-not-found', 'parent-self'])
def parent_error(request):
    return request.param


@pytest.fixture(params=['id', 'doi'])
def record_ident(request):
    return request.param


@pytest.fixture(params=[None, 'parent_id'])  # todo: this can be expanded
def record_list_filter(request):
    return request.param


def assert_db_state(records):
    """Verify that the DB record table contains the given record batch."""
    result = TestSession.execute(select(Record)).scalars().all()
    result.sort(key=lambda r: r.id)
    records.sort(key=lambda r: r.id)
    assert len(result) == len(records)
    for n, row in enumerate(result):
        assert row.id == records[n].id
        assert row.doi == records[n].doi
        assert row.sid == records[n].sid
        assert row.metadata_ == records[n].metadata_
        assert_new_timestamp(row.timestamp)
        assert row.collection_id == records[n].collection_id
        assert row.schema_id == records[n].schema_id
        assert row.schema_type == records[n].schema_type
        assert row.parent_id == records[n].parent_id


def assert_audit_log(command, record, grant_type):
    result = TestSession.execute(select(RecordAudit)).scalar_one()
    assert result.client_id == 'odp.test.client'
    assert result.user_id == ('odp.test.user' if grant_type == 'authorization_code' else None)
    assert result.command == command
    assert_new_timestamp(result.timestamp)
    assert result._id == record.id
    assert result._doi == record.doi
    assert result._sid == record.sid
    assert result._metadata == record.metadata_
    assert result._collection_id == record.collection_id
    assert result._schema_id == record.schema_id
    assert result._parent_id == record.parent_id


def assert_no_audit_log():
    assert TestSession.execute(select(RecordAudit)).first() is None


def assert_json_record_result(response, json, record):
    """Verify that the API result matches the given record object."""
    assert response.status_code == 200
    assert json['id'] == record.id
    assert json['doi'] == record.doi
    assert json['sid'] == record.sid
    assert json['collection_id'] == record.collection_id
    assert json['collection_key'] == record.collection.key
    assert json['collection_name'] == record.collection.name
    assert json['provider_id'] == record.collection.provider_id
    assert json['provider_key'] == record.collection.provider.key
    assert json['provider_name'] == record.collection.provider.name
    assert json['schema_id'] == record.schema_id
    assert json['metadata'] == record.metadata_
    assert_new_timestamp(datetime.fromisoformat(json['timestamp']))
    assert json['parent_id'] == record.parent_id
    assert json['parent_doi'] == (record.parent.doi if record.parent_id else None)
    assert json['child_dois'] == {child.id: child.doi for child in record.children}

    json_tags = json['tags']
    db_tags = TestSession.execute(
        select(RecordTag).where(RecordTag.record_id == record.id)
    ).scalars().all() + TestSession.execute(
        select(CollectionTag).where(CollectionTag.collection_id == record.collection_id)
    ).scalars().all()
    assert len(json_tags) == len(db_tags)
    json_tags.sort(key=lambda t: t['tag_id'])
    db_tags.sort(key=lambda t: t.tag_id)
    for n, json_tag in enumerate(json_tags):
        assert json_tag['tag_id'] == db_tags[n].tag_id
        assert json_tag['user_id'] == db_tags[n].user_id
        assert json_tag['user_name'] == db_tags[n].user.name
        assert json_tag['data'] == db_tags[n].data
        assert_new_timestamp(datetime.fromisoformat(json_tag['timestamp']))


def assert_json_record_results(response, json, records):
    """Verify that the API result list matches the given record batch."""
    items = json['items']
    assert json['total'] == len(items) == len(records)
    items.sort(key=lambda i: i['id'])
    records.sort(key=lambda r: r.id)
    for n, record in enumerate(records):
        assert_json_record_result(response, items[n], record)


@pytest.mark.require_scope(ODPScope.RECORD_READ)
def test_list_records(api, record_batch, scopes, collection_constraint, record_list_filter):
    authorized = ODPScope.RECORD_READ in scopes

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
        expected_result_batch = record_batch
    elif collection_constraint == 'collection_match':
        authorized_collections = [r.collection for r in record_batch[1:3]]
        expected_result_batch = record_batch[1:3]
    elif collection_constraint == 'collection_mismatch':
        authorized_collections = [CollectionFactory()]
        expected_result_batch = []

    params = {}
    if record_list_filter == 'parent_id':
        params |= dict(
            parent_id=(parent_id := record_batch[0].id)
        )
        expected_result_batch = list(filter(
            lambda rec: rec.parent_id == parent_id,
            expected_result_batch
        ))

    r = api(scopes, user_collections=authorized_collections).get('/record/', params=params)

    if authorized:
        assert_json_record_results(r, r.json(), expected_result_batch)
    else:
        assert_forbidden(r)

    assert_db_state(record_batch)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.RECORD_READ)
def test_get_record(api, record_batch, scopes, collection_constraint, record_ident):
    authorized = ODPScope.RECORD_READ in scopes and collection_constraint in ('collection_any', 'collection_match')

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    elif collection_constraint == 'collection_match':
        authorized_collections = [r.collection for r in record_batch[1:3]]
    elif collection_constraint == 'collection_mismatch':
        authorized_collections = [r.collection for r in record_batch[0:2]]

    if record_ident == 'id':
        r = api(scopes, user_collections=authorized_collections).get(f'/record/{record_batch[2].id}')
    elif record_ident == 'doi':
        if not (doi := record_batch[2].doi):
            return
        r = api(scopes, user_collections=authorized_collections).get(f'/record/doi/{doi.swapcase()}')  # case-insensitive DOI retrieval

    if authorized:
        assert_json_record_result(r, r.json(), record_batch[2])
    else:
        assert_forbidden(r)

    assert_db_state(record_batch)
    assert_no_audit_log()


def test_get_record_not_found(api, record_batch, collection_constraint, record_ident):
    scopes = [ODPScope.RECORD_READ]

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    else:
        authorized_collections = [r.collection for r in record_batch[1:3]]

    route = f'/record/{uuid.uuid4()}' if record_ident == 'id' else '/record/doi/10.55555/foo'
    r = api(scopes, user_collections=authorized_collections).get(route)

    assert_not_found(r)
    assert_db_state(record_batch)
    assert_no_audit_log()


@pytest.mark.parametrize('admin_route, scopes, collection_tags', [
    (False, [ODPScope.RECORD_WRITE], []),
    (False, [ODPScope.RECORD_WRITE], [ODPCollectionTag.FROZEN]),
    (False, [ODPScope.RECORD_WRITE], [ODPCollectionTag.PUBLISHED]),
    (False, [ODPScope.RECORD_WRITE], [ODPCollectionTag.FROZEN, ODPCollectionTag.PUBLISHED]),
    (False, [], []),
    (False, all_scopes, []),
    (False, all_scopes_excluding(ODPScope.RECORD_WRITE), []),
    (True, [ODPScope.RECORD_ADMIN], []),
    (True, [ODPScope.RECORD_ADMIN], [ODPCollectionTag.FROZEN, ODPCollectionTag.PUBLISHED]),
    (True, [], []),
    (True, all_scopes, []),
    (True, all_scopes_excluding(ODPScope.RECORD_ADMIN), []),
])
def test_create_record(api, record_batch, admin_route, scopes, collection_tags, collection_constraint, with_parent):
    route = '/record/admin/' if admin_route else '/record/'

    authorized = admin_route and ODPScope.RECORD_ADMIN in scopes or \
                 not admin_route and ODPScope.RECORD_WRITE in scopes
    authorized = authorized and collection_constraint in ('collection_any', 'collection_match')

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    elif collection_constraint == 'collection_match':
        authorized_collections = [r.collection for r in record_batch[1:3]]
    elif collection_constraint == 'collection_mismatch':
        authorized_collections = [r.collection for r in record_batch[0:2]]

    if collection_constraint in ('collection_match', 'collection_mismatch'):
        new_record_collection = record_batch[2].collection
    else:
        new_record_collection = None  # new collection

    if with_parent == 'doi':
        parent_doi = record_batch[0].doi.swapcase()  # ensure DOI ref works case-insensitively
    elif with_parent == 'doi.org':
        parent_doi = f'https://doi.org/{record_batch[0].doi.swapcase()}'
    else:
        parent_doi = None

    modified_record_batch = record_batch + [record := record_build(
        collection=new_record_collection,
        collection_tags=collection_tags,
        parent_doi=parent_doi,
    )]

    r = api(scopes, user_collections=authorized_collections).post(route, json=dict(
        doi=record.doi,
        sid=record.sid,
        collection_id=record.collection_id,
        schema_id=record.schema_id,
        metadata=record.metadata_,
    ))

    if authorized:
        if not admin_route and ODPCollectionTag.FROZEN in collection_tags:
            assert_unprocessable(r, 'A record cannot be added to a frozen collection')
            assert_db_state(record_batch)
            assert_no_audit_log()
        else:
            record.id = r.json().get('id')
            if record.doi and parent_doi:
                record.parent = record_batch[0]
                record.parent_id = record_batch[0].id
            assert_json_record_result(r, r.json(), record)
            assert_db_state(modified_record_batch)
            assert_audit_log('insert', record, api.grant_type)
    else:
        assert_forbidden(r)
        assert_db_state(record_batch)
        assert_no_audit_log()


def test_create_record_conflict(api, record_batch_with_ids, is_admin_route, collection_constraint, ident_conflict):
    route = '/record/admin/' if is_admin_route else '/record/'
    scopes = [ODPScope.RECORD_ADMIN] if is_admin_route else [ODPScope.RECORD_WRITE]
    authorized = collection_constraint in ('collection_any', 'collection_match')

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    elif collection_constraint == 'collection_match':
        authorized_collections = [r.collection for r in record_batch_with_ids[1:3]]
    elif collection_constraint == 'collection_mismatch':
        authorized_collections = [r.collection for r in record_batch_with_ids[0:2]]

    if collection_constraint in ('collection_match', 'collection_mismatch'):
        new_record_collection = record_batch_with_ids[2].collection
    else:
        new_record_collection = None  # new collection

    if ident_conflict == 'doi':
        record = record_build(
            doi=record_batch_with_ids[0].doi.swapcase(),  # check for case-insensitive DOI collision
            collection=new_record_collection,
        )
    elif ident_conflict == 'sid':
        record = record_build(
            sid=record_batch_with_ids[0].sid,
            collection=new_record_collection,
        )
    else:
        record = record_build(
            doi=record_batch_with_ids[0].doi,
            sid=record_batch_with_ids[1].sid,
            collection=new_record_collection,
        )

    r = api(scopes, user_collections=authorized_collections).post(route, json=dict(
        doi=record.doi,
        sid=record.sid,
        collection_id=record.collection_id,
        schema_id=record.schema_id,
        metadata=record.metadata_,
    ))

    if authorized:
        if ident_conflict in ('doi', 'both'):
            assert_conflict(r, 'DOI is already in use')
        else:
            assert_conflict(r, 'SID is already in use')
    else:
        assert_forbidden(r)

    assert_db_state(record_batch_with_ids)
    assert_no_audit_log()


def test_create_or_update_record_parent_error(api, record_batch, create_or_update, is_admin_route, collection_constraint, parent_error):
    route = '/record/admin/' if is_admin_route else '/record/'
    scopes = [ODPScope.RECORD_ADMIN] if is_admin_route else [ODPScope.RECORD_WRITE]
    authorized = collection_constraint in ('collection_any', 'collection_match')

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    elif collection_constraint == 'collection_match':
        authorized_collections = [r.collection for r in record_batch[1:3]]
    elif collection_constraint == 'collection_mismatch':
        authorized_collections = [r.collection for r in record_batch[0:2]]

    if collection_constraint in ('collection_match', 'collection_mismatch'):
        collection = record_batch[2].collection
    else:
        collection = None  # new collection

    kwargs = dict(
        collection=collection,
        identifiers='doi',
    )
    if create_or_update == 'update':
        kwargs |= dict(
            id=record_batch[2].id,
        )

    if parent_error == 'non-doi':
        record = record_build(
            **kwargs,
            parent_doi='foo',
        )
    elif parent_error == 'multiple-parents':
        record = record_build(
            **kwargs,
            parent_doi='10.55555/foo',
        )
        record.metadata_["relatedIdentifiers"] += [{
            "relatedIdentifier": "10.55555/bar",
            "relatedIdentifierType": "DOI",
            "relationType": "IsPartOf"
        }]
    elif parent_error == 'parent-not-found':
        record = record_build(
            **kwargs,
            parent_doi='10.55555/foo',
        )
    elif parent_error == 'parent-self':
        record = record_build(
            **kwargs,
        )
        record.metadata_["relatedIdentifiers"] = [{
            "relatedIdentifier": record.doi,
            "relatedIdentifierType": "DOI",
            "relationType": "IsPartOf"
        }]

    client = api(scopes, user_collections=authorized_collections)

    if create_or_update == 'create':
        func = client.post
    elif create_or_update == 'update':
        func = client.put
        route += record.id

    r = func(route, json=dict(
        doi=record.doi,
        collection_id=record.collection_id,
        schema_id=record.schema_id,
        metadata=record.metadata_,
    ))

    if authorized:
        if parent_error == 'non-doi':
            assert_unprocessable(r, 'Parent reference is not a valid DOI.')
        elif parent_error == 'multiple-parents':
            assert_unprocessable(r, 'Cannot determine parent DOI: found multiple related identifiers with relation IsPartOf and type DOI.')
        elif parent_error == 'parent-not-found':
            assert_unprocessable(r, 'Record not found for parent DOI 10.55555/foo')
        elif parent_error == 'parent-self':
            assert_unprocessable(r, 'DOI cannot be a parent of itself.')
    else:
        assert_forbidden(r)

    assert_db_state(record_batch)
    assert_no_audit_log()


@pytest.mark.parametrize('admin_route, scopes, collection_tags', [
    (False, [ODPScope.RECORD_WRITE], []),
    (False, [ODPScope.RECORD_WRITE], [ODPCollectionTag.FROZEN]),
    (False, [ODPScope.RECORD_WRITE], [ODPCollectionTag.FROZEN, ODPCollectionTag.HARVESTED]),
    (False, [ODPScope.RECORD_WRITE], [ODPCollectionTag.PUBLISHED]),
    (False, [ODPScope.RECORD_WRITE], [ODPCollectionTag.PUBLISHED, ODPCollectionTag.HARVESTED]),
    (False, [ODPScope.RECORD_WRITE], [ODPCollectionTag.FROZEN, ODPCollectionTag.PUBLISHED]),
    (False, [ODPScope.RECORD_WRITE], [ODPCollectionTag.FROZEN, ODPCollectionTag.PUBLISHED, ODPCollectionTag.HARVESTED]),
    (False, [], []),
    (False, all_scopes, []),
    (False, all_scopes_excluding(ODPScope.RECORD_WRITE), []),
    (True, [ODPScope.RECORD_ADMIN], []),
    (True, [ODPScope.RECORD_ADMIN], [ODPCollectionTag.FROZEN, ODPCollectionTag.PUBLISHED]),
    (True, [ODPScope.RECORD_ADMIN], [ODPCollectionTag.FROZEN, ODPCollectionTag.PUBLISHED, ODPCollectionTag.HARVESTED]),
    (True, [], []),
    (True, all_scopes, []),
    (True, all_scopes_excluding(ODPScope.RECORD_ADMIN), []),
])
def test_update_record(api, record_batch, admin_route, scopes, collection_tags, collection_constraint, with_parent):
    route = '/record/admin/' if admin_route else '/record/'

    authorized = admin_route and ODPScope.RECORD_ADMIN in scopes or \
                 not admin_route and ODPScope.RECORD_WRITE in scopes
    authorized = authorized and collection_constraint in ('collection_any', 'collection_match')

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    elif collection_constraint == 'collection_match':
        authorized_collections = [r.collection for r in record_batch[1:3]]
    elif collection_constraint == 'collection_mismatch':
        authorized_collections = [r.collection for r in record_batch[0:1]]

    if collection_constraint in ('collection_match', 'collection_mismatch'):
        modified_record_collection = record_batch[1].collection  # switch collection
    else:
        modified_record_collection = None  # new collection

    if with_parent == 'doi':
        parent_doi = record_batch[0].doi.swapcase()  # ensure DOI ref works case-insensitively
    elif with_parent == 'doi.org':
        parent_doi = f'https://doi.org/{record_batch[0].doi.swapcase()}'
    else:
        parent_doi = None

    modified_record_batch = record_batch.copy()
    modified_record_batch[2] = (record := record_build(
        id=record_batch[2].id,
        doi=record_batch[2].doi,
        collection=modified_record_collection,
        collection_tags=collection_tags,
        parent_doi=parent_doi,
    ))

    r = api(scopes, user_collections=authorized_collections).put(route + record.id, json=dict(
        doi=record.doi,
        sid=record.sid,
        collection_id=record.collection_id,
        schema_id=record.schema_id,
        metadata=record.metadata_,
    ))

    if authorized:
        if not admin_route and ODPCollectionTag.FROZEN in collection_tags:
            assert_unprocessable(r, 'Cannot update a record belonging to a frozen collection')
            assert_db_state(record_batch)
            assert_no_audit_log()
        else:
            if record.doi and parent_doi:
                record.parent = record_batch[0]
                record.parent_id = record_batch[0].id
            assert_json_record_result(r, r.json(), record)
            assert_db_state(modified_record_batch)
            assert_audit_log('update', record, api.grant_type)
    else:
        assert_forbidden(r)
        assert_db_state(record_batch)
        assert_no_audit_log()


def test_update_record_not_found(api, record_batch, is_admin_route, collection_constraint):
    # if not found on the admin route, the record is created!
    route = '/record/admin/' if is_admin_route else '/record/'
    scopes = [ODPScope.RECORD_ADMIN] if is_admin_route else [ODPScope.RECORD_WRITE]
    authorized = collection_constraint in ('collection_any', 'collection_match')

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    elif collection_constraint == 'collection_match':
        authorized_collections = [r.collection for r in record_batch[1:3]]
    elif collection_constraint == 'collection_mismatch':
        authorized_collections = [r.collection for r in record_batch[0:2]]

    if collection_constraint in ('collection_match', 'collection_mismatch'):
        collection = record_batch[2].collection
    else:
        collection = None  # new collection

    modified_record_batch = record_batch + [record := record_build(
        id=str(uuid.uuid4()),
        collection=collection,
    )]

    r = api(scopes, user_collections=authorized_collections).put(route + record.id, json=dict(
        doi=record.doi,
        sid=record.sid,
        collection_id=record.collection_id,
        schema_id=record.schema_id,
        metadata=record.metadata_,
    ))

    if authorized:
        if is_admin_route:
            assert_json_record_result(r, r.json(), record)
            assert_db_state(modified_record_batch)
            assert_audit_log('insert', record, api.grant_type)
        else:
            assert_not_found(r)
            assert_db_state(record_batch)
            assert_no_audit_log()
    else:
        assert_forbidden(r)
        assert_db_state(record_batch)
        assert_no_audit_log()


def test_update_record_conflict(api, record_batch_with_ids, is_admin_route, collection_constraint, ident_conflict):
    route = '/record/admin/' if is_admin_route else '/record/'
    scopes = [ODPScope.RECORD_ADMIN] if is_admin_route else [ODPScope.RECORD_WRITE]
    authorized = collection_constraint in ('collection_any', 'collection_match')

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    elif collection_constraint == 'collection_match':
        authorized_collections = [r.collection for r in record_batch_with_ids[1:3]]
    elif collection_constraint == 'collection_mismatch':
        authorized_collections = [r.collection for r in record_batch_with_ids[0:1]]

    if collection_constraint in ('collection_match', 'collection_mismatch'):
        modified_record_collection = record_batch_with_ids[1].collection  # switch collection
    else:
        modified_record_collection = None  # new collection

    if ident_conflict == 'doi':
        record = record_build(
            id=record_batch_with_ids[2].id,
            doi=record_batch_with_ids[0].doi.swapcase(),  # check for case-insensitive DOI collision
            collection=modified_record_collection,
        )
    elif ident_conflict == 'sid':
        record = record_build(
            id=record_batch_with_ids[2].id,
            sid=record_batch_with_ids[0].sid,
            collection=modified_record_collection,
        )
    else:
        record = record_build(
            id=record_batch_with_ids[2].id,
            doi=record_batch_with_ids[0].doi,
            sid=record_batch_with_ids[1].sid,
            collection=modified_record_collection,
        )

    r = api(scopes, user_collections=authorized_collections).put(route + record.id, json=dict(
        doi=record.doi,
        sid=record.sid,
        collection_id=record.collection_id,
        schema_id=record.schema_id,
        metadata=record.metadata_,
    ))

    if authorized:
        if ident_conflict in ('doi', 'both'):
            assert_conflict(r, 'DOI is already in use')
        else:
            assert_conflict(r, 'SID is already in use')
    else:
        assert_forbidden(r)

    assert_db_state(record_batch_with_ids)
    assert_no_audit_log()


def test_update_record_doi_change(api, record_batch_with_ids, is_admin_route, collection_constraint, doi_change, is_published_record):
    route = '/record/admin/' if is_admin_route else '/record/'
    scopes = [ODPScope.RECORD_ADMIN] if is_admin_route else [ODPScope.RECORD_WRITE]
    authorized = collection_constraint in ('collection_any', 'collection_match')

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    elif collection_constraint == 'collection_match':
        authorized_collections = [r.collection for r in record_batch_with_ids[1:3]]
    elif collection_constraint == 'collection_mismatch':
        authorized_collections = [r.collection for r in record_batch_with_ids[0:1]]

    if collection_constraint in ('collection_match', 'collection_mismatch'):
        modified_record_collection = record_batch_with_ids[1].collection  # switch collection
    else:
        modified_record_collection = None  # new collection

    if is_published_record:
        FactorySession.add(PublishedRecord(
            id=record_batch_with_ids[2].id,
            doi=record_batch_with_ids[2].doi if is_published_record == 'doi' else None,
        ))
        FactorySession.commit()

    modified_record_batch = record_batch_with_ids.copy()
    if doi_change == 'change':
        modified_record_batch[2] = (record := record_build(
            identifiers='doi',
            id=record_batch_with_ids[2].id,
            collection=modified_record_collection,
        ))
    elif doi_change == 'remove':
        modified_record_batch[2] = (record := record_build(
            identifiers='sid',
            id=record_batch_with_ids[2].id,
            collection=modified_record_collection,
        ))

    r = api(scopes, user_collections=authorized_collections).put(route + record.id, json=dict(
        doi=record.doi,
        sid=record.sid,
        collection_id=record.collection_id,
        schema_id=record.schema_id,
        metadata=record.metadata_,
    ))

    if authorized:
        if is_published_record == 'doi':
            assert_unprocessable(r, 'The DOI has been published and cannot be modified.')
            assert_db_state(record_batch_with_ids)
            assert_no_audit_log()
        else:
            assert_json_record_result(r, r.json(), record)
            assert_db_state(modified_record_batch)
            assert_audit_log('update', record, api.grant_type)
    else:
        assert_forbidden(r)
        assert_db_state(record_batch_with_ids)
        assert_no_audit_log()


@pytest.mark.parametrize('admin_route, scopes, collection_tags', [
    (False, [ODPScope.RECORD_WRITE], []),
    (False, [ODPScope.RECORD_WRITE], [ODPCollectionTag.FROZEN]),
    (False, [ODPScope.RECORD_WRITE], [ODPCollectionTag.PUBLISHED]),
    (False, [ODPScope.RECORD_WRITE], [ODPCollectionTag.FROZEN, ODPCollectionTag.PUBLISHED]),
    (False, [], []),
    (False, all_scopes, []),
    (False, all_scopes_excluding(ODPScope.RECORD_WRITE), []),
    (True, [ODPScope.RECORD_ADMIN], []),
    (True, [ODPScope.RECORD_ADMIN], [ODPCollectionTag.FROZEN, ODPCollectionTag.PUBLISHED]),
    (True, [], []),
    (True, all_scopes, []),
    (True, all_scopes_excluding(ODPScope.RECORD_ADMIN), []),
])
def test_delete_record(api, record_batch_with_ids, admin_route, scopes, collection_tags, collection_constraint, is_published_record):
    route = '/record/admin/' if admin_route else '/record/'

    authorized = admin_route and ODPScope.RECORD_ADMIN in scopes or \
                 not admin_route and ODPScope.RECORD_WRITE in scopes
    authorized = authorized and collection_constraint in ('collection_any', 'collection_match')

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    elif collection_constraint == 'collection_match':
        authorized_collections = [r.collection for r in record_batch_with_ids[1:3]]
    elif collection_constraint == 'collection_mismatch':
        authorized_collections = [r.collection for r in record_batch_with_ids[0:2]]

    for ct in collection_tags:
        CollectionTagFactory(
            collection=record_batch_with_ids[2].collection,
            tag=TagFactory(id=ct, type='collection'),
        )

    if is_published_record:
        FactorySession.add(PublishedRecord(
            id=record_batch_with_ids[2].id,
            doi=record_batch_with_ids[2].doi if is_published_record == 'doi' else None,
        ))
        FactorySession.commit()

    modified_record_batch = record_batch_with_ids.copy()
    deleted_record = modified_record_batch[2]
    del modified_record_batch[2]

    r = api(scopes, user_collections=authorized_collections).delete(f'{route}{(record_id := record_batch_with_ids[2].id)}')

    if authorized:
        if not admin_route and ODPCollectionTag.FROZEN in collection_tags:
            assert_unprocessable(r, 'Cannot delete a record belonging to a frozen collection')
            assert_db_state(record_batch_with_ids)
            assert_no_audit_log()
        elif is_published_record:
            assert_unprocessable(r, 'The record has been published and cannot be deleted. Please retract the record instead.')
            assert_db_state(record_batch_with_ids)
            assert_no_audit_log()
        else:
            assert_ok_null(r)
            assert_db_state(modified_record_batch)
            assert_audit_log('delete', deleted_record, api.grant_type)
    else:
        assert_forbidden(r)
        assert_db_state(record_batch_with_ids)
        assert_no_audit_log()


def test_delete_record_not_found(api, record_batch, is_admin_route, collection_constraint):
    route = '/record/admin/' if is_admin_route else '/record/'
    scopes = [ODPScope.RECORD_ADMIN] if is_admin_route else [ODPScope.RECORD_WRITE]

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    else:
        authorized_collections = [r.collection for r in record_batch[1:3]]

    r = api(scopes, user_collections=authorized_collections).delete(f'{route}foo')

    assert_not_found(r)
    assert_db_state(record_batch)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.RECORD_QC)
def test_tag_record(api, record_batch_no_tags, scopes, collection_constraint, tag_cardinality):
    authorized = ODPScope.RECORD_QC in scopes and collection_constraint in ('collection_any', 'collection_match')

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    elif collection_constraint == 'collection_match':
        authorized_collections = [r.collection for r in record_batch_no_tags[1:3]]
    elif collection_constraint == 'collection_mismatch':
        authorized_collections = [r.collection for r in record_batch_no_tags[0:2]]

    client = api(scopes, user_collections=authorized_collections)
    tag = new_generic_tag('record', tag_cardinality)

    # TAG 1
    r = client.post(
        f'/record/{(record_id := record_batch_no_tags[2].id)}/tag',
        json=(record_tag_1 := dict(
            tag_id=tag.id,
            data={'comment': 'test1'},
            cardinality=tag_cardinality,
            public=tag.public,
        ) | keyword_tag_args(tag.vocabulary, 0)))

    if authorized:
        assert_tag_instance_output(r, record_tag_1, api.grant_type)
        assert_tag_instance_db_state('record', api.grant_type, record_id, record_tag_1)
        assert_tag_instance_audit_log(
            'record', api.grant_type,
            dict(command='insert', object_id=record_id, tag_instance=record_tag_1),
        )

        # TAG 2
        r = client.post(
            f'/record/{(record_id := record_batch_no_tags[2].id)}/tag',
            json=(record_tag_2 := dict(
                tag_id=tag.id,
                data=record_tag_1['data'] if tag.vocabulary else {'comment': 'test2'},
                cardinality=tag_cardinality,
                public=tag.public,
            ) | keyword_tag_args(tag.vocabulary, 1)))

        assert_tag_instance_output(r, record_tag_2, api.grant_type)
        if tag_cardinality in ('one', 'user'):
            assert_tag_instance_db_state('record', api.grant_type, record_id, record_tag_2)
            assert_tag_instance_audit_log(
                'record', api.grant_type,
                dict(command='insert', object_id=record_id, tag_instance=record_tag_1),
                dict(command='update', object_id=record_id, tag_instance=record_tag_2),
            )
        elif tag_cardinality == 'multi':
            assert_tag_instance_db_state('record', api.grant_type, record_id, record_tag_1, record_tag_2)
            assert_tag_instance_audit_log(
                'record', api.grant_type,
                dict(command='insert', object_id=record_id, tag_instance=record_tag_1),
                dict(command='insert', object_id=record_id, tag_instance=record_tag_2),
            )

        # TAG 3 - different client/user
        client = api(
            scopes,
            user_collections=authorized_collections,
            client_id='testclient2',
            role_id='testrole2',
            user_id='testuser2',
            user_email='test2@saeon.ac.za',
        )
        r = client.post(
            f'/record/{(record_id := record_batch_no_tags[2].id)}/tag',
            json=(record_tag_3 := dict(
                tag_id=tag.id,
                data=record_tag_1['data'] if tag.vocabulary else {'comment': 'test3'},
                cardinality=tag_cardinality,
                public=tag.public,
                auth_client_id='testclient2',
                auth_user_id='testuser2' if api.grant_type == 'authorization_code' else None,
                user_id='testuser2' if api.grant_type == 'authorization_code' else None,
                user_email='test2@saeon.ac.za' if api.grant_type == 'authorization_code' else None,
            ) | keyword_tag_args(tag.vocabulary, 2)))

        assert_tag_instance_output(r, record_tag_3, api.grant_type)
        if tag_cardinality == 'one':
            assert_tag_instance_db_state('record', api.grant_type, record_id, record_tag_3)
            assert_tag_instance_audit_log(
                'record', api.grant_type,
                dict(command='insert', object_id=record_id, tag_instance=record_tag_1),
                dict(command='update', object_id=record_id, tag_instance=record_tag_2),
                dict(command='update', object_id=record_id, tag_instance=record_tag_3),
            )
        elif tag_cardinality == 'user':
            if api.grant_type == 'client_credentials':
                # user_id is null so it's an update
                record_tags = (record_tag_3,)
                tag3_command = 'update'
            else:
                record_tags = (record_tag_2, record_tag_3,)
                tag3_command = 'insert'

            assert_tag_instance_db_state('record', api.grant_type, record_id, *record_tags)
            assert_tag_instance_audit_log(
                'record', api.grant_type,
                dict(command='insert', object_id=record_id, tag_instance=record_tag_1),
                dict(command='update', object_id=record_id, tag_instance=record_tag_2),
                dict(command=tag3_command, object_id=record_id, tag_instance=record_tag_3),
            )
        elif tag_cardinality == 'multi':
            assert_tag_instance_db_state('record', api.grant_type, record_id, record_tag_1, record_tag_2, record_tag_3)
            assert_tag_instance_audit_log(
                'record', api.grant_type,
                dict(command='insert', object_id=record_id, tag_instance=record_tag_1),
                dict(command='insert', object_id=record_id, tag_instance=record_tag_2),
                dict(command='insert', object_id=record_id, tag_instance=record_tag_3),
            )

    else:
        assert_forbidden(r)
        assert_tag_instance_db_state('record', api.grant_type, record_id)
        assert_tag_instance_audit_log_empty('record')

    assert_db_state(record_batch_no_tags)
    assert_no_audit_log()


@pytest.mark.parametrize('admin_route, scopes', [
    (False, [ODPScope.RECORD_QC]),
    (False, []),
    (False, all_scopes),
    (False, all_scopes_excluding(ODPScope.RECORD_QC)),
    (True, [ODPScope.RECORD_ADMIN]),
    (True, []),
    (True, all_scopes),
    (True, all_scopes_excluding(ODPScope.RECORD_ADMIN)),
])
def test_untag_record(api, record_batch_no_tags, admin_route, scopes, collection_constraint, is_same_user):
    route = '/record/admin/' if admin_route else '/record/'

    authorized = admin_route and ODPScope.RECORD_ADMIN in scopes or \
                 not admin_route and ODPScope.RECORD_QC in scopes
    authorized = authorized and collection_constraint in ('collection_any', 'collection_match')

    try_skip_collection_constraint(api.grant_type, collection_constraint)

    if collection_constraint == 'collection_any':
        authorized_collections = None  # => all
    elif collection_constraint == 'collection_match':
        authorized_collections = [r.collection for r in record_batch_no_tags[1:3]]
    elif collection_constraint == 'collection_mismatch':
        authorized_collections = [r.collection for r in record_batch_no_tags[0:2]]

    client = api(scopes, user_collections=authorized_collections)
    record = record_batch_no_tags[2]
    record_tags = RecordTagFactory.create_batch(randint(1, 3), record=record)

    tag = new_generic_tag('record')
    if is_same_user:
        record_tag_1 = RecordTagFactory(
            record=record,
            tag=tag,
            keyword=tag.vocabulary.keywords[2] if tag.vocabulary else None,
            user=FactorySession.get(User, 'odp.test.user') if api.grant_type == 'authorization_code' else None,
        )
    else:
        record_tag_1 = RecordTagFactory(
            record=record,
            tag=tag,
            keyword=tag.vocabulary.keywords[2] if tag.vocabulary else None,
        )
    record_tag_1_dict = {
        'tag_id': record_tag_1.tag_id,
        'user_id': record_tag_1.user_id,
        'data': record_tag_1.data,
        'keyword_id': record_tag_1.keyword_id,
    }

    r = client.delete(f'{route}{record.id}/tag/{record_tag_1.id}')

    if authorized:
        if not admin_route and not is_same_user:
            assert_forbidden(r)
            assert_tag_instance_db_state('record', api.grant_type, record.id, *record_tags, record_tag_1)
            assert_tag_instance_audit_log_empty('record')
        else:
            assert_ok_null(r)
            assert_tag_instance_db_state('record', api.grant_type, record.id, *record_tags)
            assert_tag_instance_audit_log(
                'record', api.grant_type,
                dict(command='delete', object_id=record.id, tag_instance=record_tag_1_dict),
            )
    else:
        assert_forbidden(r)
        assert_tag_instance_db_state('record', api.grant_type, record.id, *record_tags, record_tag_1)
        assert_tag_instance_audit_log_empty('record')

    assert_db_state(record_batch_no_tags)
    assert_no_audit_log()
