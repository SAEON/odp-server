from datetime import date, datetime
from random import randint

import pytest
from sqlalchemy import select

from odp.const import ODPDateRangeIncType, ODPPackageTag, ODPScope, ODPTagSchema
from odp.db.models import Package, PackageAudit, PackageTag, Resource, Scope, Tag, User
from test import TestSession
from test.api import all_scopes, test_resource
from test.api.assertions import (
    assert_forbidden,
    assert_method_not_allowed,
    assert_new_timestamp,
    assert_not_found,
    assert_ok_null,
    assert_unprocessable,
)
from test.api.assertions.tags import (
    assert_tag_instance_audit_log,
    assert_tag_instance_audit_log_empty,
    assert_tag_instance_db_state,
    assert_tag_instance_output,
    keyword_tag_args,
    new_generic_tag,
)
from test.api.conftest import try_skip_user_provider_constraint
from test.factories import (
    FactorySession,
    PackageFactory,
    PackageTagFactory,
    ProviderFactory,
    ResourceFactory,
    SchemaFactory,
    TagFactory,
)


@pytest.fixture
def package_batch(request):
    """Create and commit a batch of Package instances, with
    associated resources."""
    with_tags = request.node.get_closest_marker('package_batch_with_tags') is not None
    package_2_no_resources = request.node.get_closest_marker('package_2_no_resources') is not None

    packages = PackageFactory.create_batch(randint(3, 5))
    for n, package in enumerate(packages):
        if n == 2 and package_2_no_resources:
            resources = []
        else:
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
        assert row.status == packages[n].status
        assert_new_timestamp(row.timestamp)
        assert row.provider_id == packages[n].provider_id
        assert row.metadata_ == packages[n].metadata_
        assert row.schema_id == packages[n].schema_id
        assert row.schema_type == packages[n].schema_type

    result = TestSession.execute(select(Resource.package_id, Resource.id)).all()
    result.sort(key=lambda r: (r.package_id, r.id))
    package_resources = []
    for package in packages:
        for resource_id in package.resource_ids:
            package_resources += [(package.id, resource_id)]
    package_resources.sort()
    assert result == package_resources


def assert_audit_log(command, package, grant_type):
    result = TestSession.execute(select(PackageAudit)).scalar_one()
    assert result.client_id == 'odp.test.client'
    assert result.user_id == ('odp.test.user' if grant_type == 'authorization_code' else None)
    assert result.command == command
    assert_new_timestamp(result.timestamp)
    assert result._id == package.id
    assert result._key == package.key
    assert result._status == package.status
    assert result._provider_id == package.provider_id
    assert result._schema_id == package.schema_id
    assert sorted(result._resources) == sorted(package.resource_ids)


def assert_no_audit_log():
    assert TestSession.execute(select(PackageAudit)).first() is None


def assert_json_result(response, json, package, detail=False, old_provider_key=None):
    """Verify that the API result matches the given package object."""
    # todo: check linked record
    assert response.status_code == 200
    assert json['id'] == package.id

    # Numeric suffix will differ between factory- and API-generated package key;
    # update the factory object to match API output. Pass in old_provider_key for
    # updates, since the package key cannot change.
    date = datetime.now().strftime('%Y_%m_%d')
    assert json['key'].startswith(f'{old_provider_key or package.provider.key}_{date}_')
    package.key = json['key']

    assert json['status'] == package.status
    assert_new_timestamp(datetime.fromisoformat(json['timestamp']))
    assert json['provider_id'] == package.provider_id
    assert json['provider_key'] == package.provider.key
    assert sorted(json['resource_ids']) == sorted(package.resource_ids)
    assert json['schema_id'] == package.schema_id
    assert json['schema_uri'] == package.schema.uri

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

    if detail:
        assert json['metadata'] == package.metadata_


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
        provider_id=package.provider_id,
        schema_id=package.schema_id,
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


def test_update_package(api):
    r = api(all_scopes).put('/package/foo')
    assert_method_not_allowed(r)
    assert_no_audit_log()


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

    package_provider = package_batch[2].provider if package_new_provider == 'same' else ProviderFactory()
    package = package_build(
        id=package_batch[2].id,
        key=package_batch[2].key,
        status=package_batch[2].status,
        provider=package_provider,
        schema_id='SAEON.DataCite4' if package_batch[2].schema_id == 'SAEON.ISO19115' else 'SAEON.ISO19115',
        metadata_=package_batch[2].metadata_,
        resource_ids=[resource.id for resource in package_batch[2].resources],
    )
    old_provider_key = package_batch[2].provider.key

    r = api(scopes, **api_kwargs).put(f'/package/admin/{package.id}', json=dict(
        provider_id=package.provider_id,
        schema_id=package.schema_id,
    ))

    if authorized:
        assert_json_result(r, r.json(), package, detail=True, old_provider_key=old_provider_key)
        assert_db_state(package_batch[:2] + [package] + package_batch[3:])
        assert_audit_log('update', package, api.grant_type)
    else:
        assert_forbidden(r)
        assert_db_state(package_batch)
        assert_no_audit_log()


def test_admin_update_package_not_found(
        api,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
):
    scopes = [ODPScope.PACKAGE_ADMIN]
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    package = package_build(id='foo')

    r = api(scopes, **api_kwargs).put(f'/package/admin/{package.id}', json=dict(
        provider_id=package.provider_id,
        schema_id=package.schema_id,
    ))

    assert_not_found(r)
    assert_db_state(package_batch)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.PACKAGE_WRITE)
@pytest.mark.parametrize('error', [None, 'not_found', 'has_resources', 'has_record'])
@pytest.mark.package_2_no_resources
def test_delete_package(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
        error,
):
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    authorized_scope = ODPScope.PACKAGE_WRITE in scopes
    authorized_constraint = (
            client_provider_constraint in ('client_provider_any', 'client_provider_match') and
            (api.grant_type == 'client_credentials' or user_provider_constraint == 'user_provider_match')
    )

    _test_delete_package(
        api,
        scopes,
        package_batch,
        '/package/',
        authorized_scope,
        authorized_constraint,
        api_kwargs,
        error,
    )


@pytest.mark.require_scope(ODPScope.PACKAGE_ADMIN)
@pytest.mark.parametrize('error', [None, 'not_found', 'has_resources', 'has_record'])
@pytest.mark.package_2_no_resources
def test_admin_delete_package(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
        error,
):
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )

    _test_delete_package(
        api,
        scopes,
        package_batch,
        '/package/admin/',
        ODPScope.PACKAGE_ADMIN in scopes,
        True,
        api_kwargs,
        error,
    )


def _test_delete_package(
        api,
        scopes,
        package_batch,
        route,
        authorized_scope,
        authorized_constraint,
        api_kwargs,
        error,
):
    if error == 'not_found':
        deleted_package_id = 'foo'
    elif error == 'has_resources':
        resources = ResourceFactory.create_batch(randint(1, 4), package=package_batch[2])
        package_batch[2].resource_ids = [resource.id for resource in resources]
        deleted_package_id = package_batch[2].id
    elif error == 'has_record':
        # TODO record-package linkage
        pytest.skip('TODO')
    else:
        deleted_package = PackageFactory.stub(
            id=package_batch[2].id,
            key=package_batch[2].key,
            status=package_batch[2].status,
            provider_id=package_batch[2].provider_id,
            schema_id=package_batch[2].schema_id,
            resource_ids=[],
        )
        deleted_package_id = deleted_package.id

    deleted_status = package_batch[2].status

    r = api(scopes, **api_kwargs).delete(f'{route}{deleted_package_id}')

    changed = False
    if not authorized_scope:
        assert_forbidden(r)
    elif error == 'not_found':
        assert_not_found(r)
    elif not authorized_constraint:
        assert_forbidden(r)
    elif 'admin' not in route and deleted_status != 'pending':
        assert_unprocessable(r, "Package status must be 'pending'")
    elif error in ('has_resources', 'has_record'):
        assert_unprocessable(r, 'A package with an associated record or resources cannot be deleted.')
    else:
        assert_ok_null(r)
        assert_db_state(package_batch[:2] + package_batch[3:])
        assert_audit_log('delete', deleted_package, api.grant_type)
        changed = True

    if not changed:
        assert_db_state(package_batch)
        assert_no_audit_log()


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
    tag = new_generic_tag('package', tag_cardinality)

    # TAG 1
    r = client.post(
        f'/package/{(package_id := package_batch[2].id)}/tag',
        json=(package_tag_1 := dict(
            tag_id=tag.id,
            data={'comment': 'test1'},
            cardinality=tag_cardinality,
            public=tag.public,
        ) | keyword_tag_args(tag.vocabulary, 0)))

    if authorized and package_batch[2].status == 'pending':
        assert_tag_instance_output(r, package_tag_1, api.grant_type)
        assert_tag_instance_db_state('package', api.grant_type, package_id, package_tag_1)
        assert_tag_instance_audit_log(
            'package', api.grant_type,
            dict(command='insert', object_id=package_id, tag_instance=package_tag_1),
        )

        # TAG 2
        r = client.post(
            f'/package/{(package_id := package_batch[2].id)}/tag',
            json=(package_tag_2 := dict(
                tag_id=tag.id,
                data=package_tag_1['data'] if tag.vocabulary else {'comment': 'test2'},
                cardinality=tag_cardinality,
                public=tag.public,
            ) | keyword_tag_args(tag.vocabulary, 1)))

        assert_tag_instance_output(r, package_tag_2, api.grant_type)
        if tag_cardinality in ('one', 'user'):
            assert_tag_instance_db_state('package', api.grant_type, package_id, package_tag_2)
            assert_tag_instance_audit_log(
                'package', api.grant_type,
                dict(command='insert', object_id=package_id, tag_instance=package_tag_1),
                dict(command='update', object_id=package_id, tag_instance=package_tag_2),
            )
        elif tag_cardinality == 'multi':
            assert_tag_instance_db_state('package', api.grant_type, package_id, package_tag_1, package_tag_2)
            assert_tag_instance_audit_log(
                'package', api.grant_type,
                dict(command='insert', object_id=package_id, tag_instance=package_tag_1),
                dict(command='insert', object_id=package_id, tag_instance=package_tag_2),
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
                data=package_tag_1['data'] if tag.vocabulary else {'comment': 'test3'},
                cardinality=tag_cardinality,
                public=tag.public,
                auth_client_id='testclient2',
                auth_user_id='testuser2' if api.grant_type == 'authorization_code' else None,
                user_id='testuser2' if api.grant_type == 'authorization_code' else None,
                user_email='test2@saeon.ac.za' if api.grant_type == 'authorization_code' else None,
            ) | keyword_tag_args(tag.vocabulary, 2)))

        assert_tag_instance_output(r, package_tag_3, api.grant_type)
        if tag_cardinality == 'one':
            assert_tag_instance_db_state('package', api.grant_type, package_id, package_tag_3)
            assert_tag_instance_audit_log(
                'package', api.grant_type,
                dict(command='insert', object_id=package_id, tag_instance=package_tag_1),
                dict(command='update', object_id=package_id, tag_instance=package_tag_2),
                dict(command='update', object_id=package_id, tag_instance=package_tag_3),
            )
        elif tag_cardinality == 'user':
            if api.grant_type == 'client_credentials':
                # user_id is null so it's an update
                package_tags = (package_tag_3,)
                tag3_command = 'update'
            else:
                package_tags = (package_tag_2, package_tag_3,)
                tag3_command = 'insert'

            assert_tag_instance_db_state('package', api.grant_type, package_id, *package_tags)
            assert_tag_instance_audit_log(
                'package', api.grant_type,
                dict(command='insert', object_id=package_id, tag_instance=package_tag_1),
                dict(command='update', object_id=package_id, tag_instance=package_tag_2),
                dict(command=tag3_command, object_id=package_id, tag_instance=package_tag_3),
            )
        elif tag_cardinality == 'multi':
            assert_tag_instance_db_state('package', api.grant_type, package_id, package_tag_1, package_tag_2,
                                         package_tag_3)
            assert_tag_instance_audit_log(
                'package', api.grant_type,
                dict(command='insert', object_id=package_id, tag_instance=package_tag_1),
                dict(command='insert', object_id=package_id, tag_instance=package_tag_2),
                dict(command='insert', object_id=package_id, tag_instance=package_tag_3),
            )

    else:
        if not authorized:
            assert_forbidden(r)
        else:  # package_batch[2].status != 'pending'
            assert_unprocessable(r, "Package status must be 'pending'")

        assert_tag_instance_db_state('package', api.grant_type, package_id)
        assert_tag_instance_audit_log_empty('package')

    assert_db_state(package_batch)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.PACKAGE_DOI)
@pytest.mark.parametrize('same_user', [True, False])
def test_untag_package(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
        same_user,
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

    _test_untag_package(
        api,
        api_kwargs,
        scopes,
        authorized,
        package_batch,
        same_user,
        False,
    )


@pytest.mark.require_scope(ODPScope.PACKAGE_ADMIN)
@pytest.mark.parametrize('same_user', [True, False])
def test_admin_untag_package(
        api,
        scopes,
        package_batch,
        client_provider_constraint,
        user_provider_constraint,
        same_user,
):
    api_kwargs = parameterize_api_fixture(
        package_batch,
        api.grant_type,
        client_provider_constraint,
        user_provider_constraint,
    )
    authorized = ODPScope.PACKAGE_ADMIN in scopes

    _test_untag_package(
        api,
        api_kwargs,
        scopes,
        authorized,
        package_batch,
        same_user,
        True,
    )


def _test_untag_package(
        api,
        api_kwargs,
        scopes,
        authorized,
        package_batch,
        same_user,
        admin_route,
):
    client = api(scopes, **api_kwargs)
    route = '/package/admin/' if admin_route else '/package/'

    package = package_batch[2]
    package_tags = PackageTagFactory.create_batch(randint(1, 3), package=package)

    tag = new_generic_tag('package')
    if same_user:
        package_tag_1 = PackageTagFactory(
            package=package,
            tag=tag,
            keyword=tag.vocabulary.keywords[2] if tag.vocabulary else None,
            user=FactorySession.get(User, 'odp.test.user') if api.grant_type == 'authorization_code' else None,
        )
    else:
        package_tag_1 = PackageTagFactory(
            package=package,
            tag=tag,
            keyword=tag.vocabulary.keywords[2] if tag.vocabulary else None,
        )
    package_tag_1_dict = {
        'tag_id': package_tag_1.tag_id,
        'user_id': package_tag_1.user_id,
        'data': package_tag_1.data,
        'keyword_id': package_tag_1.keyword_id,
    }

    r = client.delete(f'{route}{package.id}/tag/{package_tag_1.id}')

    if authorized:
        if not admin_route and package.status != 'pending':
            assert_unprocessable(r, "Package status must be 'pending'")
            assert_tag_instance_db_state('package', api.grant_type, package.id, *package_tags, package_tag_1)
            assert_tag_instance_audit_log_empty('package')
        elif not admin_route and not same_user:
            assert_forbidden(r)
            assert_tag_instance_db_state('package', api.grant_type, package.id, *package_tags, package_tag_1)
            assert_tag_instance_audit_log_empty('package')
        else:
            assert_ok_null(r)
            assert_tag_instance_db_state('package', api.grant_type, package.id, *package_tags)
            assert_tag_instance_audit_log(
                'package', api.grant_type,
                dict(command='delete', object_id=package.id, tag_instance=package_tag_1_dict),
            )
    else:
        assert_forbidden(r)
        assert_tag_instance_db_state('package', api.grant_type, package.id, *package_tags, package_tag_1)
        assert_tag_instance_audit_log_empty('package')

    assert_db_state(package_batch)
    assert_no_audit_log()


def get_date_range_tag(schema_id: str, schema_uri: str, tag_id: str):
    date_range_schema = SchemaFactory(
        id=schema_id,
        uri=schema_uri,
        type='tag',
    )

    return TagFactory(
        id=tag_id,
        type='package',
        cardinality='one',
        scope=FactorySession.get(Scope, (ODPScope.PACKAGE_WRITE, 'odp')),
        schema=date_range_schema
    )


def test_inc_end_date():
    from odp.package.date_range import DateRangeInc

    test_package = PackageFactory()

    PackageTagFactory(
        package=test_package,
        tag=get_date_range_tag(
            ODPTagSchema.DATERANGEINC,
            'https://odp.saeon.ac.za/schema/tag/daterangeinc',
            ODPPackageTag.DATERANGEINC
        ),
        data={
            'end': ODPDateRangeIncType.CURRENT_DATE
        }
    )

    PackageTagFactory(
        package=test_package,
        tag=get_date_range_tag(
            ODPTagSchema.DATERANGE,
            'https://odp.saeon.ac.za/schema/tag/daterange',
            ODPPackageTag.DATERANGE
        ),
        data={
            'start': '1990/01/01',
            'end': '2000/01/01'
        }
    )

    DateRangeInc().execute()

    stmt = (
        select(Package, PackageTag)
        .where(PackageTag.package_id == test_package.id)
        .where(PackageTag.tag_id == ODPPackageTag.DATERANGE.value)
    )

    res = TestSession.execute(stmt).first()

    assert res.PackageTag.data['end'] == date.today().isoformat()
