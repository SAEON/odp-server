from datetime import datetime, date
from random import randint

import pytest
from sqlalchemy import select

from odp.const import ODPScope, ODPTagSchema, ODPDateRangeIncType, ODPPackageTag
from odp.const.db import ScopeType
from odp.db.models import Package, PackageAudit, PackageTag, PackageTagAudit, Resource, Scope, Tag, User
from test import TestSession
from test.api import assert_empty_result, assert_forbidden, assert_new_timestamp, assert_not_found, \
    assert_tag_instance_output, test_resource
from test.api.conftest import try_skip_user_provider_constraint
from test.factories import FactorySession, PackageFactory, PackageTagFactory, ProviderFactory, ResourceFactory, \
    SchemaFactory, TagFactory, ScopeFactory


@pytest.fixture
def package_batch(request):
    """Create and commit a batch of Package instances, with
    associated resources."""
    with_tags = request.node.get_closest_marker('package_batch_with_tags') is not None

    packages = PackageFactory.create_batch(randint(3, 5))
    for package in packages:
        resources = ResourceFactory.create_batch(randint(0, 4), package=package)
        package.resource_ids = [resource.id for resource in resources]
        if with_tags:
            PackageTagFactory.create_batch(randint(0, 3), package=package)

    return packages


def package_build(provider=None, resource_ids=None, **id):
    """Build and return an uncommitted Package instance.
    Referenced provider is however committed."""
    package = PackageFactory.build(
        **id,
        provider=(provider := provider or ProviderFactory()),
        provider_id=provider.id,
    )
    package.resource_ids = resource_ids or []
    return package


def assert_db_state(packages):
    """Verify that the package table contains the given package batch,
    and that the resource table contains the associated resource references."""
    result = TestSession.execute(select(Package)).scalars().all()
    result.sort(key=lambda p: p.id)
    packages.sort(key=lambda p: p.id)
    assert len(result) == len(packages)
    for n, row in enumerate(result):
        assert row.id == packages[n].id
        assert row.key == packages[n].key
        assert row.title == packages[n].title
        assert row.status == packages[n].status
        assert_new_timestamp(row.timestamp)
        assert row.provider_id == packages[n].provider_id

    result = TestSession.execute(select(Resource.package_id, Resource.id)).all()
    result.sort(key=lambda r: (r.package_id, r.id))
    package_resources = []
    for package in packages:
        for resource_id in package.resource_ids:
            package_resources += [(package.id, resource_id)]
    package_resources.sort()
    assert result == package_resources


def assert_db_tag_state(package_id, grant_type, *package_tags):
    """Verify that the package_tag table contains the given package tags."""
    result = TestSession.execute(select(PackageTag)).scalars().all()
    result.sort(key=lambda r: r.timestamp)

    assert len(result) == len(package_tags)
    for n, row in enumerate(result):
        assert row.package_id == package_id
        assert row.tag_type == 'package'
        assert_new_timestamp(row.timestamp)
        if isinstance(package_tag := package_tags[n], PackageTag):
            assert row.tag_id == package_tag.tag_id
            assert row.user_id == package_tag.user_id
            assert row.data == package_tag.data
        else:
            user_id = package_tag.get('user_id', 'odp.test.user')
            assert row.tag_id == package_tag['tag_id']
            assert row.user_id == (user_id if grant_type == 'authorization_code' else None)
            assert row.data == package_tag['data']


def assert_audit_log(command, package, grant_type):
    result = TestSession.execute(select(PackageAudit)).scalar_one()
    assert result.client_id == 'odp.test.client'
    assert result.user_id == ('odp.test.user' if grant_type == 'authorization_code' else None)
    assert result.command == command
    assert_new_timestamp(result.timestamp)
    assert result._id == package.id
    assert result._key == package.key
    assert result._title == package.title
    assert result._status == package.status
    assert result._provider_id == package.provider_id
    assert sorted(result._resources) == sorted(package.resource_ids)


def assert_no_audit_log():
    assert TestSession.execute(select(PackageAudit)).first() is None


def assert_tag_audit_log(grant_type, *entries):
    result = TestSession.execute(select(PackageTagAudit)).scalars().all()
    assert len(result) == len(entries)
    for n, row in enumerate(result):
        auth_client_id = entries[n]['package_tag'].get('auth_client_id', 'odp.test.client')
        auth_user_id = entries[n]['package_tag'].get('auth_user_id',
                                                     'odp.test.user' if grant_type == 'authorization_code' else None)
        assert row.client_id == auth_client_id
        assert row.user_id == auth_user_id
        assert row.command == entries[n]['command']
        assert_new_timestamp(row.timestamp)
        assert row._package_id == entries[n]['package_id']
        assert row._tag_id == entries[n]['package_tag']['tag_id']
        assert row._user_id == entries[n]['package_tag'].get('user_id', auth_user_id)
        assert row._data == entries[n]['package_tag']['data']


def assert_no_tag_audit_log():
    assert TestSession.execute(select(PackageTagAudit)).first() is None


def assert_json_result(response, json, package, detail=False):
    """Verify that the API result matches the given package object."""
    # todo: check linked record
    assert response.status_code == 200
    assert json['id'] == package.id
    assert json['key'] == package.key
    assert json['title'] == package.title
    assert json['status'] == package.status
    assert_new_timestamp(datetime.fromisoformat(json['timestamp']))
    assert json['provider_id'] == package.provider_id
    assert json['provider_key'] == package.provider.key
    assert sorted(json['resource_ids']) == sorted(package.resource_ids)

    if detail:
        json_resources = json['resources']
        db_resources = TestSession.execute(
            select(Resource).where(Resource.package_id == package.id)
        ).scalars().all()
        assert len(json_resources) == len(db_resources)
        json_resources.sort(key=lambda r: r['id'])
        db_resources.sort(key=lambda r: r.id)
        for n, json_resource in enumerate(json_resources):
            db_resources[n].archive_paths = {}  # stub for attr used locally in test_resource
            test_resource.assert_json_result(response, json_resource, db_resources[n])

        json_tags = json['tags']
        db_tags = TestSession.execute(
            select(PackageTag, Tag, User).join(Tag).join(User).where(PackageTag.package_id == package.id)
        ).all()
        assert len(json_tags) == len(db_tags)
        json_tags.sort(key=lambda t: t['id'])
        db_tags.sort(key=lambda t: t.PackageTag.id)
        for n, json_tag in enumerate(json_tags):
            assert json_tag['tag_id'] == db_tags[n].PackageTag.tag_id
            assert json_tag['user_id'] == db_tags[n].PackageTag.user_id
            assert json_tag['user_name'] == db_tags[n].User.name
            assert json_tag['data'] == db_tags[n].PackageTag.data
            assert_new_timestamp(db_tags[n].PackageTag.timestamp)
            assert json_tag['cardinality'] == db_tags[n].Tag.cardinality
            assert json_tag['public'] == db_tags[n].Tag.public


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
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.PACKAGE_READ_ALL)
def test_list_all_packages(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
):
    """Configured as for test_list_packages, but for this scope+endpoint
    all packages can always be read unconditionally."""
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
        force_mismatch=True,
    )
    authorized = ODPScope.PACKAGE_READ_ALL in scopes
    expected_result_batch = package_batch

    # todo: test provider_id filter
    r = api(scopes, **api_kwargs).get('/package/all/')

    if authorized:
        assert_json_results(r, r.json(), expected_result_batch)
    else:
        assert_forbidden(r)

    assert_db_state(package_batch)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.PACKAGE_READ)
@pytest.mark.package_batch_with_tags
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
        assert_json_result(r, r.json(), package_batch[2], detail=True)
    else:
        assert_forbidden(r)

    assert_db_state(package_batch)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.PACKAGE_READ_ALL)
@pytest.mark.package_batch_with_tags
def test_get_any_package(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
):
    """Configured as for test_get_package, but for this scope+endpoint
    any package can always be read unconditionally."""
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    authorized = ODPScope.PACKAGE_READ_ALL in scopes

    r = api(scopes, **api_kwargs).get(f'/package/all/{package_batch[2].id}')

    if authorized:
        assert_json_result(r, r.json(), package_batch[2], detail=True)
    else:
        assert_forbidden(r)

    assert_db_state(package_batch)
    assert_no_audit_log()


@pytest.mark.parametrize('route', ['/package/', '/package/all/'])
def test_get_package_not_found(
        api,
        route,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
):
    scopes = [ODPScope.PACKAGE_READ_ALL] if 'all' in route else [ODPScope.PACKAGE_READ]
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    r = api(scopes, **api_kwargs).get(f'{route}foo')
    assert_not_found(r)
    assert_db_state(package_batch)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.PACKAGE_WRITE)
def test_create_package(
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
            ODPScope.PACKAGE_WRITE in scopes and
            client_provider_constraint in ('client_provider_any', 'client_provider_match') and
            (api.grant_type == 'client_credentials' or user_provider_constraint == 'user_provider_match')
    )

    _test_create_package(
        api,
        scopes,
        package_batch,
        '/package/',
        authorized,
        api_kwargs,
    )


@pytest.mark.require_scope(ODPScope.PACKAGE_ADMIN)
def test_admin_create_package(
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
    authorized = ODPScope.PACKAGE_ADMIN in scopes

    _test_create_package(
        api,
        scopes,
        package_batch,
        '/package/admin/',
        authorized,
        api_kwargs,
    )


def _test_create_package(
        api,
        scopes,
        package_batch,
        route,
        authorized,
        api_kwargs,
):
    package = package_build(
        status='pending',
        provider=package_batch[2].provider,
    )

    r = api(scopes, **api_kwargs).post(route, json=dict(
        title=package.title,
        provider_id=package.provider_id,
    ))

    if authorized:
        package.id = r.json().get('id')
        assert_json_result(r, r.json(), package, detail=True)
        assert_db_state(package_batch + [package])
        assert_audit_log('insert', package, api.grant_type)
    else:
        assert_forbidden(r)
        assert_db_state(package_batch)
        assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.PACKAGE_WRITE)
@pytest.mark.parametrize('package_new_provider', ['same', 'different'])
@pytest.mark.package_batch_with_tags
def test_update_package(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
        package_new_provider,
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
        authorized = authorized and package_new_provider == 'same'

    _test_update_package(
        api,
        scopes,
        package_batch,
        package_new_provider,
        '/package/',
        authorized,
        api_kwargs,
    )


@pytest.mark.require_scope(ODPScope.PACKAGE_ADMIN)
@pytest.mark.parametrize('package_new_provider', ['same', 'different'])
@pytest.mark.package_batch_with_tags
def test_admin_update_package(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
        package_new_provider,
):
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    authorized = ODPScope.PACKAGE_ADMIN in scopes

    _test_update_package(
        api,
        scopes,
        package_batch,
        package_new_provider,
        '/package/admin/',
        authorized,
        api_kwargs,
    )


def _test_update_package(
        api,
        scopes,
        package_batch,
        package_new_provider,
        route,
        authorized,
        api_kwargs,
):
    package_provider = package_batch[2].provider if package_new_provider == 'same' else ProviderFactory()
    package_build_args = dict(
        id=package_batch[2].id,
        status=package_batch[2].status,
        provider=package_provider,
    )
    if package_batch[2].resources:
        # key must stay the same if package #2 has any resources
        package_build_args |= dict(
            key=package_batch[2].key,
            resource_ids=[resource.id for resource in package_batch[2].resources],
        )

    package = package_build(**package_build_args)

    r = api(scopes, **api_kwargs).put(f'{route}{package.id}', json=dict(
        title=package.title,
        provider_id=package.provider_id,
    ))

    if authorized:
        assert_json_result(r, r.json(), package, detail=True)
        assert_db_state(package_batch[:2] + [package] + package_batch[3:])
        assert_audit_log('update', package, api.grant_type)
    else:
        assert_forbidden(r)
        assert_db_state(package_batch)
        assert_no_audit_log()


@pytest.mark.parametrize('route', ['/package/', '/package/admin/'])
def test_update_package_not_found(
        api,
        route,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
):
    scopes = [ODPScope.PACKAGE_ADMIN] if 'admin' in route else [ODPScope.PACKAGE_WRITE]
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    package = package_build(id='foo')

    r = api(scopes, **api_kwargs).put(f'{route}{package.id}', json=dict(
        title=package.title,
        provider_id=package.provider_id,
    ))

    assert_not_found(r)
    assert_db_state(package_batch)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.PACKAGE_WRITE)
def test_delete_package(
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
            ODPScope.PACKAGE_WRITE in scopes and
            client_provider_constraint in ('client_provider_any', 'client_provider_match') and
            (api.grant_type == 'client_credentials' or user_provider_constraint == 'user_provider_match')
    )

    _test_delete_package(
        api,
        scopes,
        package_batch,
        '/package/',
        authorized,
        api_kwargs,
    )


@pytest.mark.require_scope(ODPScope.PACKAGE_ADMIN)
def test_admin_delete_package(
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
    authorized = ODPScope.PACKAGE_ADMIN in scopes

    _test_delete_package(
        api,
        scopes,
        package_batch,
        '/package/admin/',
        authorized,
        api_kwargs,
    )


def _test_delete_package(
        api,
        scopes,
        package_batch,
        route,
        authorized,
        api_kwargs,
):
    deleted_package = package_batch[2]

    r = api(scopes, **api_kwargs).delete(f'{route}{deleted_package.id}')

    if authorized:
        assert_empty_result(r)
        assert_db_state(package_batch[:2] + package_batch[3:])
        assert_audit_log('delete', deleted_package, api.grant_type)
    else:
        assert_forbidden(r)
        assert_db_state(package_batch)
        assert_no_audit_log()


@pytest.mark.parametrize('route', ['/package/', '/package/admin/'])
def test_delete_package_not_found(
        api,
        route,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
):
    scopes = [ODPScope.PACKAGE_ADMIN] if 'admin' in route else [ODPScope.PACKAGE_WRITE]
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    r = api(scopes, **api_kwargs).delete(f'{route}foo')
    assert_not_found(r)
    assert_db_state(package_batch)
    assert_no_audit_log()


def new_generic_tag(cardinality):
    # we can use any scope; just make it something other than PACKAGE_ADMIN
    return TagFactory(
        type='package',
        cardinality=cardinality,
        scope=FactorySession.get(Scope, (ODPScope.PACKAGE_DOI, ScopeType.odp)),
        schema=SchemaFactory(type='tag', uri='https://odp.saeon.ac.za/schema/tag/generic'),
    )


@pytest.mark.require_scope(ODPScope.PACKAGE_DOI)
def test_tag_package(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
        tag_cardinality,
):
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    authorized = (
            ODPScope.PACKAGE_DOI in scopes and
            client_provider_constraint in ('client_provider_any', 'client_provider_match') and
            (api.grant_type == 'client_credentials' or user_provider_constraint == 'user_provider_match')
    )

    client = api(scopes, **api_kwargs)
    tag = new_generic_tag(tag_cardinality)

    # TAG 1
    r = client.post(
        f'/package/{(package_id := package_batch[2].id)}/tag',
        json=(package_tag_1 := dict(
            tag_id=tag.id,
            data={'comment': 'test1'},
            cardinality=tag_cardinality,
            public=tag.public,
        )))

    if authorized:
        assert_tag_instance_output(r, package_tag_1, api.grant_type)
        assert_db_tag_state(package_id, api.grant_type, package_tag_1)
        assert_tag_audit_log(
            api.grant_type,
            dict(command='insert', package_id=package_id, package_tag=package_tag_1),
        )

        # TAG 2
        r = client.post(
            f'/package/{(package_id := package_batch[2].id)}/tag',
            json=(package_tag_2 := dict(
                tag_id=tag.id,
                data={'comment': 'test2'},
                cardinality=tag_cardinality,
                public=tag.public,
            )))

        assert_tag_instance_output(r, package_tag_2, api.grant_type)
        if tag_cardinality in ('one', 'user'):
            assert_db_tag_state(package_id, api.grant_type, package_tag_2)
            assert_tag_audit_log(
                api.grant_type,
                dict(command='insert', package_id=package_id, package_tag=package_tag_1),
                dict(command='update', package_id=package_id, package_tag=package_tag_2),
            )
        elif tag_cardinality == 'multi':
            assert_db_tag_state(package_id, api.grant_type, package_tag_1, package_tag_2)
            assert_tag_audit_log(
                api.grant_type,
                dict(command='insert', package_id=package_id, package_tag=package_tag_1),
                dict(command='insert', package_id=package_id, package_tag=package_tag_2),
            )

        # TAG 3 - different client/user
        client = api(
            scopes,
            **api_kwargs,
            client_id='testclient2',
            role_id='testrole2',
            user_id='testuser2',
            user_email='test2@saeon.ac.za',
        )
        r = client.post(
            f'/package/{(package_id := package_batch[2].id)}/tag',
            json=(package_tag_3 := dict(
                tag_id=tag.id,
                data={'comment': 'test3'},
                cardinality=tag_cardinality,
                public=tag.public,
                auth_client_id='testclient2',
                auth_user_id='testuser2' if api.grant_type == 'authorization_code' else None,
                user_id='testuser2' if api.grant_type == 'authorization_code' else None,
                user_email='test2@saeon.ac.za' if api.grant_type == 'authorization_code' else None,
            )))

        assert_tag_instance_output(r, package_tag_3, api.grant_type)
        if tag_cardinality == 'one':
            assert_db_tag_state(package_id, api.grant_type, package_tag_3)
            assert_tag_audit_log(
                api.grant_type,
                dict(command='insert', package_id=package_id, package_tag=package_tag_1),
                dict(command='update', package_id=package_id, package_tag=package_tag_2),
                dict(command='update', package_id=package_id, package_tag=package_tag_3),
            )
        elif tag_cardinality == 'user':
            if api.grant_type == 'client_credentials':
                # user_id is null so it's an update
                package_tags = (package_tag_3,)
                tag3_command = 'update'
            else:
                package_tags = (package_tag_2, package_tag_3,)
                tag3_command = 'insert'

            assert_db_tag_state(package_id, api.grant_type, *package_tags)
            assert_tag_audit_log(
                api.grant_type,
                dict(command='insert', package_id=package_id, package_tag=package_tag_1),
                dict(command='update', package_id=package_id, package_tag=package_tag_2),
                dict(command=tag3_command, package_id=package_id, package_tag=package_tag_3),
            )
        elif tag_cardinality == 'multi':
            assert_db_tag_state(package_id, api.grant_type, package_tag_1, package_tag_2, package_tag_3)
            assert_tag_audit_log(
                api.grant_type,
                dict(command='insert', package_id=package_id, package_tag=package_tag_1),
                dict(command='insert', package_id=package_id, package_tag=package_tag_2),
                dict(command='insert', package_id=package_id, package_tag=package_tag_3),
            )

    else:  # not authorized
        assert_forbidden(r)
        assert_db_tag_state(package_id, api.grant_type)
        assert_no_tag_audit_log()

    assert_db_state(package_batch)
    assert_no_audit_log()


def date_range_package_tag(package: PackageFactory, package_write_scope: ScopeFactory):
    schema_uri = 'https://odp.saeon.ac.za/schema/tag/daterange'
    return PackageTagFactory(
        package=package,
        tag=get_date_range_tag(ODPTagSchema.DATERANGE, schema_uri, ODPPackageTag.DATERANGE, package_write_scope),
        data={
            'start': '1990/01/01',
            'end': '2000/01/01'
        }
    )


def date_range_inc_package_tag(package: PackageFactory, package_write_scope: ScopeFactory):
    schema_uri = 'https://odp.saeon.ac.za/schema/tag/daterangeinc'
    return PackageTagFactory(
        package=package,
        tag=get_date_range_tag(ODPTagSchema.DATERANGEINC, schema_uri, ODPPackageTag.DATERANGEINC, package_write_scope),
        data={
            'end': ODPDateRangeIncType.CURRENT_DATE
        }
    )


def get_date_range_tag(schema_id: str, schema_uri: str, tag_id: str, package_write_scope: ScopeFactory):
    date_range_schema = SchemaFactory(
        id=schema_id,
        uri=schema_uri,
        type='tag',
    )

    return TagFactory(
        id=tag_id,
        type='package',
        cardinality='one',
        scope=package_write_scope,
        schema=date_range_schema
    )


def test_inc_end_date():
    from odp.package.date_range import DateRangeInc

    test_package = PackageFactory()

    package_write_scope = ScopeFactory(
        id=ODPScope.PACKAGE_WRITE,
        type='odp'
    )

    date_range_inc_package_tag(test_package, package_write_scope)
    date_range_package_tag(test_package, package_write_scope)

    DateRangeInc().execute()

    stmt = (
        select(Package, PackageTag)
        .where(PackageTag.package_id == test_package.id)
        .where(PackageTag.tag_id == ODPPackageTag.DATERANGE.value)
    )

    res = TestSession.execute(stmt).first()

    assert res.PackageTag.data['end'] == date.today().strftime("%Y/%m/%d")
