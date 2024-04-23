from datetime import datetime
from random import randint

import pytest
from sqlalchemy import select

from odp.const import ODPScope
from odp.db.models import Package, PackageResource
from test import TestSession
from test.api import assert_empty_result, assert_forbidden, assert_new_timestamp, assert_not_found
from test.api.conftest import try_skip_user_provider_constraint
from test.factories import FactorySession, PackageFactory, ProviderFactory, ResourceFactory


@pytest.fixture
def package_batch():
    """Create and commit a batch of Package instances, with
    associated resources. The #2 package's resources get the
    same provider as the package."""
    packages = []
    for n in range(randint(3, 5)):
        packages += [package := PackageFactory(
            provider=(provider := ProviderFactory()),
            resources=(resources := ResourceFactory.create_batch(
                randint(n == 2, 4),  # at least one resource if n == 2
                provider=provider if n == 2 else ProviderFactory(),
            )),
        )]
        package.resource_ids = [resource.id for resource in resources]

    return packages


def package_build(package_provider=None, resource_provider=None, **id):
    """Build and return an uncommitted Package instance.
    Referenced providers and resources are however committed."""
    package_provider = package_provider or ProviderFactory()
    resource_provider = resource_provider or ProviderFactory()
    package = PackageFactory.build(
        **id,
        provider=package_provider,
        provider_id=package_provider.id,
        resources=(resources := ResourceFactory.create_batch(randint(1, 4), provider=resource_provider)),
    )
    package.resource_ids = [resource.id for resource in resources]
    return package


def assert_db_state(packages):
    """Verify that the package table contains the given package batch,
    and that the package_resource table contains the associated resource
    references."""
    result = TestSession.execute(select(Package)).scalars().all()
    result.sort(key=lambda p: p.id)
    packages.sort(key=lambda p: p.id)
    assert len(result) == len(packages)
    for n, row in enumerate(result):
        assert row.id == packages[n].id
        assert row.metadata_ == packages[n].metadata_
        assert row.notes == packages[n].notes
        assert_new_timestamp(row.timestamp)
        assert row.provider_id == packages[n].provider_id
        assert row.schema_id == packages[n].schema_id
        assert row.schema_type == packages[n].schema_type == 'metadata'

    result = TestSession.execute(select(PackageResource.package_id, PackageResource.resource_id)).all()
    result.sort(key=lambda pr: (pr.package_id, pr.resource_id))
    package_resources = []
    for package in packages:
        for resource_id in package.resource_ids:
            package_resources += [(package.id, resource_id)]
    package_resources.sort()
    assert result == package_resources


def assert_json_result(response, json, package):
    """Verify that the API result matches the given package object."""
    # todo: check linked record
    assert response.status_code == 200
    assert json['id'] == package.id
    assert json['provider_id'] == package.provider_id
    assert json['provider_key'] == package.provider.key
    assert json['schema_id'] == package.schema_id
    assert json['metadata'] == package.metadata_
    assert json['notes'] == package.notes
    assert_new_timestamp(datetime.fromisoformat(json['timestamp']))
    assert sorted(json['resource_ids']) == sorted(package.resource_ids)


def assert_json_results(response, json, packages):
    """Verify that the API result list matches the given package batch."""
    items = json['items']
    assert json['total'] == len(items) == len(packages)
    items.sort(key=lambda i: i['id'])
    packages.sort(key=lambda p: p.id)
    for n, package in enumerate(packages):
        assert_json_result(response, items[n], package)


def parameterize_api_fixture(
        packages,
        grant_type,
        client_provider_constraint,
        user_provider_constraint,
        force_mismatch=False,
):
    """Return tuple(client_provider, user_providers) for parameterizing
    the api fixture, based on constraint params and generated package batch.

    Set force_mismatch=True for the list test; this creates a new provider
    for the mismatch cases. For all the other tests we can reuse any existing
    providers other than the #2 package's provider for the mismatches.
    """
    try_skip_user_provider_constraint(grant_type, user_provider_constraint)

    if client_provider_constraint == 'client_provider_any':
        client_provider = None
    elif client_provider_constraint == 'client_provider_match':
        client_provider = packages[2].provider
    elif client_provider_constraint == 'client_provider_mismatch':
        client_provider = ProviderFactory() if force_mismatch else packages[0].provider

    if user_provider_constraint == 'user_provider_none':
        user_providers = None
    elif user_provider_constraint == 'user_provider_match':
        user_providers = [p.provider for p in packages[1:3]]
    elif user_provider_constraint == 'user_provider_mismatch':
        user_providers = [ProviderFactory()] if force_mismatch else [p.provider for p in packages[0:2]]

    return dict(client_provider=client_provider, user_providers=user_providers)


@pytest.mark.require_scope(ODPScope.PACKAGE_READ)
def test_list_packages(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
):
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
        force_mismatch=True,
    )
    authorized = ODPScope.PACKAGE_READ in scopes

    if client_provider_constraint == 'client_provider_any':
        expected_result_batch = package_batch
    elif client_provider_constraint == 'client_provider_match':
        expected_result_batch = [package_batch[2]]
    elif client_provider_constraint == 'client_provider_mismatch':
        expected_result_batch = []

    if api.grant_type == 'authorization_code':
        if user_provider_constraint == 'user_provider_match':
            expected_result_batch = list(set(package_batch[1:3]).intersection(expected_result_batch))
        else:
            expected_result_batch = []

    # todo: test provider_id filter
    r = api(scopes, **api_kwargs).get('/package/')

    if authorized:
        assert_json_results(r, r.json(), expected_result_batch)
    else:
        assert_forbidden(r)

    assert_db_state(package_batch)


@pytest.mark.require_scope(ODPScope.PACKAGE_READ)
def test_get_package(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
):
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    authorized = (
            ODPScope.PACKAGE_READ in scopes and
            client_provider_constraint in ('client_provider_any', 'client_provider_match') and
            (api.grant_type == 'client_credentials' or user_provider_constraint == 'user_provider_match')
    )

    r = api(scopes, **api_kwargs).get(f'/package/{package_batch[2].id}')

    if authorized:
        assert_json_result(r, r.json(), package_batch[2])
    else:
        assert_forbidden(r)

    assert_db_state(package_batch)


def test_get_package_not_found(
        api,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
):
    scopes = [ODPScope.PACKAGE_READ]
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    r = api(scopes, **api_kwargs).get('/package/foo')
    assert_not_found(r)
    assert_db_state(package_batch)


@pytest.mark.require_scope(ODPScope.PACKAGE_WRITE)
@pytest.mark.parametrize('package_resource_provider', ['same', 'different'])
def test_create_package(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
        package_resource_provider,
):
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    authorized = (
            ODPScope.PACKAGE_WRITE in scopes and
            client_provider_constraint in ('client_provider_any', 'client_provider_match') and
            (api.grant_type == 'client_credentials' or user_provider_constraint == 'user_provider_match')
    )
    if (
            client_provider_constraint == 'client_provider_match' or
            user_provider_constraint == 'user_provider_match'
    ):
        authorized = authorized and package_resource_provider == 'same'

    package = package_build(
        package_provider=package_batch[2].provider,
        resource_provider=package_batch[2].provider if package_resource_provider == 'same' else None,
    )

    r = api(scopes, **api_kwargs).post('/package/', json=dict(
        provider_id=package.provider_id,
        schema_id=package.schema_id,
        metadata=package.metadata_,
        notes=package.notes,
        resource_ids=package.resource_ids,
    ))

    if authorized:
        package.id = r.json().get('id')
        assert_json_result(r, r.json(), package)
        assert_db_state(package_batch + [package])
    else:
        assert_forbidden(r)
        assert_db_state(package_batch)


@pytest.mark.require_scope(ODPScope.PACKAGE_WRITE)
@pytest.mark.parametrize('package_new_provider', ['same', 'different'])
@pytest.mark.parametrize('package_new_resource_provider', ['same', 'different'])
@pytest.mark.parametrize('package_existing_resource_provider', ['same', 'different'])
def test_update_package(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
        package_new_provider,
        package_new_resource_provider,
        package_existing_resource_provider,
):
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    authorized = (
            ODPScope.PACKAGE_WRITE in scopes and
            client_provider_constraint in ('client_provider_any', 'client_provider_match') and
            (api.grant_type == 'client_credentials' or user_provider_constraint == 'user_provider_match')
    )
    if (
            client_provider_constraint == 'client_provider_match' or
            user_provider_constraint == 'user_provider_match'
    ):
        authorized = (authorized and
                      package_new_provider == 'same' and
                      package_new_resource_provider == 'same' and
                      package_existing_resource_provider == 'same')

    if package_existing_resource_provider == 'different':
        package_batch[2].resources[0].provider = ProviderFactory()
        FactorySession.commit()

    package_provider = package_batch[2].provider if package_new_provider == 'same' else ProviderFactory()
    package = package_build(
        id=package_batch[2].id,
        package_provider=package_provider,
        resource_provider=package_provider if package_new_resource_provider == 'same' else None,
    )

    r = api(scopes, **api_kwargs).put(f'/package/{package.id}', json=dict(
        provider_id=package.provider_id,
        schema_id=package.schema_id,
        metadata=package.metadata_,
        notes=package.notes,
        resource_ids=package.resource_ids,
    ))

    if authorized:
        assert_json_result(r, r.json(), package)
        assert_db_state(package_batch[:2] + [package] + package_batch[3:])
    else:
        assert_forbidden(r)
        assert_db_state(package_batch)


def test_update_package_not_found(
        api,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
):
    scopes = [ODPScope.PACKAGE_WRITE]
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    package = package_build(id='foo')

    r = api(scopes, **api_kwargs).put(f'/package/{package.id}', json=dict(
        provider_id=package.provider_id,
        schema_id=package.schema_id,
        metadata=package.metadata_,
        notes=package.notes,
        resource_ids=package.resource_ids,
    ))

    assert_not_found(r)
    assert_db_state(package_batch)


@pytest.mark.require_scope(ODPScope.PACKAGE_WRITE)
@pytest.mark.parametrize('package_resource_provider', ['same', 'different'])
def test_delete_package(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
        package_resource_provider,
):
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    authorized = (
            ODPScope.PACKAGE_WRITE in scopes and
            client_provider_constraint in ('client_provider_any', 'client_provider_match') and
            (api.grant_type == 'client_credentials' or user_provider_constraint == 'user_provider_match')
    )
    if (
            client_provider_constraint == 'client_provider_match' or
            user_provider_constraint == 'user_provider_match'
    ):
        authorized = authorized and package_resource_provider == 'same'

    if package_resource_provider == 'different':
        package_batch[2].resources[0].provider = ProviderFactory()
        FactorySession.commit()

    r = api(scopes, **api_kwargs).delete(f'/package/{package_batch[2].id}')

    if authorized:
        assert_empty_result(r)
        assert_db_state(package_batch[:2] + package_batch[3:])
    else:
        assert_forbidden(r)
        assert_db_state(package_batch)


def test_delete_package_not_found(
        api,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
):
    scopes = [ODPScope.PACKAGE_WRITE]
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    r = api(scopes, **api_kwargs).delete('/package/foo')
    assert_not_found(r)
    assert_db_state(package_batch)