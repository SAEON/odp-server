from random import randint

import pytest
from sqlalchemy import select

from odp.const import ODPScope
from odp.db import Session
from odp.db.models import Provider
from test.api import all_scopes, all_scopes_excluding, assert_conflict, assert_empty_result, assert_forbidden, assert_not_found, assert_unprocessable
from test.factories import CollectionFactory, ProviderFactory, RecordFactory


@pytest.fixture
def provider_batch():
    """Create and commit a batch of Provider instances,
    with associated collections."""
    providers = [ProviderFactory() for _ in range(randint(3, 5))]
    for provider in providers:
        CollectionFactory.create_batch(randint(0, 3), provider=provider)
    return providers


def provider_build(**kwargs):
    """Build and return an uncommitted Provider instance."""
    return ProviderFactory.build(**kwargs)


def collection_ids(provider):
    return tuple(sorted(collection.id for collection in provider.collections))


def assert_db_state(providers):
    """Verify that the DB provider table contains the given provider batch."""
    Session.expire_all()
    result = Session.execute(select(Provider)).scalars().all()
    assert set((row.id, row.name, collection_ids(row)) for row in result) \
           == set((provider.id, provider.name, collection_ids(provider)) for provider in providers)


def assert_json_result(response, json, provider):
    """Verify that the API result matches the given provider object."""
    assert response.status_code == 200
    assert json['id'] == provider.id
    assert json['name'] == provider.name
    assert tuple(sorted(json['collection_ids'])) == collection_ids(provider)


def assert_json_results(response, json, providers):
    """Verify that the API result list matches the given provider batch."""
    items = json['items']
    assert json['total'] == len(items) == len(providers)
    items.sort(key=lambda i: i['id'])
    providers.sort(key=lambda p: p.id)
    for n, provider in enumerate(providers):
        assert_json_result(response, items[n], provider)


@pytest.mark.parametrize('scopes', [
    [ODPScope.PROVIDER_READ],
    [],
    all_scopes,
    all_scopes_excluding(ODPScope.PROVIDER_READ),
])
def test_list_providers(api, provider_batch, scopes):
    authorized = ODPScope.PROVIDER_READ in scopes
    r = api(scopes).get('/provider/')
    if authorized:
        assert_json_results(r, r.json(), provider_batch)
    else:
        assert_forbidden(r)
    assert_db_state(provider_batch)


@pytest.mark.parametrize('scopes', [
    [ODPScope.PROVIDER_READ],
    [],
    all_scopes,
    all_scopes_excluding(ODPScope.PROVIDER_READ),
])
def test_get_provider(api, provider_batch, scopes):
    authorized = ODPScope.PROVIDER_READ in scopes
    r = api(scopes).get(f'/provider/{provider_batch[2].id}')
    if authorized:
        assert_json_result(r, r.json(), provider_batch[2])
    else:
        assert_forbidden(r)
    assert_db_state(provider_batch)


def test_get_provider_not_found(api, provider_batch):
    scopes = [ODPScope.PROVIDER_READ]
    r = api(scopes).get('/provider/foo')
    assert_not_found(r)
    assert_db_state(provider_batch)


@pytest.mark.parametrize('scopes', [
    [ODPScope.PROVIDER_ADMIN],
    [],
    all_scopes,
    all_scopes_excluding(ODPScope.PROVIDER_ADMIN),
])
def test_create_provider(api, provider_batch, scopes):
    authorized = ODPScope.PROVIDER_ADMIN in scopes
    modified_provider_batch = provider_batch + [provider := provider_build()]
    r = api(scopes).post('/provider/', json=dict(
        id=provider.id,
        name=provider.name,
    ))
    if authorized:
        assert_empty_result(r)
        assert_db_state(modified_provider_batch)
    else:
        assert_forbidden(r)
        assert_db_state(provider_batch)


def test_create_provider_conflict(api, provider_batch):
    scopes = [ODPScope.PROVIDER_ADMIN]
    provider = provider_build(id=provider_batch[2].id)
    r = api(scopes).post('/provider/', json=dict(
        id=provider.id,
        name=provider.name,
    ))
    assert_conflict(r, 'Provider id is already in use')
    assert_db_state(provider_batch)


@pytest.mark.parametrize('scopes', [
    [ODPScope.PROVIDER_ADMIN],
    [],
    all_scopes,
    all_scopes_excluding(ODPScope.PROVIDER_ADMIN),
])
def test_update_provider(api, provider_batch, scopes):
    authorized = ODPScope.PROVIDER_ADMIN in scopes
    modified_provider_batch = provider_batch.copy()
    modified_provider_batch[2] = (provider := provider_build(
        id=provider_batch[2].id,
        collections=provider_batch[2].collections,
    ))
    r = api(scopes).put('/provider/', json=dict(
        id=provider.id,
        name=provider.name,
    ))
    if authorized:
        assert_empty_result(r)
        assert_db_state(modified_provider_batch)
    else:
        assert_forbidden(r)
        assert_db_state(provider_batch)


def test_update_provider_not_found(api, provider_batch):
    scopes = [ODPScope.PROVIDER_ADMIN]
    provider = provider_build(id='foo')
    r = api(scopes).put('/provider/', json=dict(
        id=provider.id,
        name=provider.name,
    ))
    assert_not_found(r)
    assert_db_state(provider_batch)


@pytest.fixture(params=[True, False])
def has_record(request):
    return request.param


@pytest.mark.parametrize('scopes', [
    [ODPScope.PROVIDER_ADMIN],
    [],
    all_scopes,
    all_scopes_excluding(ODPScope.PROVIDER_ADMIN),
])
def test_delete_provider(api, provider_batch, scopes, has_record):
    authorized = ODPScope.PROVIDER_ADMIN in scopes
    modified_provider_batch = provider_batch.copy()
    del modified_provider_batch[2]

    if has_record:
        if collection := next((c for c in provider_batch[2].collections), None):
            RecordFactory(collection=collection)
        else:
            has_record = False

    r = api(scopes).delete(f'/provider/{provider_batch[2].id}')

    if authorized:
        if has_record:
            assert_unprocessable(r, 'A provider with non-empty collections cannot be deleted.')
            assert_db_state(provider_batch)
        else:
            assert_empty_result(r)
            assert_db_state(modified_provider_batch)
    else:
        assert_forbidden(r)
        assert_db_state(provider_batch)


def test_delete_provider_not_found(api, provider_batch):
    scopes = [ODPScope.PROVIDER_ADMIN]
    r = api(scopes).delete('/provider/foo')
    assert_not_found(r)
    assert_db_state(provider_batch)
