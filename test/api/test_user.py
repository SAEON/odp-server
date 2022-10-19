from random import randint

import pytest
from sqlalchemy import select

from odp.const import ODPScope
from odp.db import Session
from odp.db.models import User
from test.api import all_scopes, all_scopes_excluding, assert_empty_result, assert_forbidden, assert_method_not_allowed, assert_not_found
from test.factories import RoleFactory, UserFactory


@pytest.fixture
def user_batch():
    """Create and commit a batch of User instances."""
    return [
        UserFactory(roles=RoleFactory.create_batch(randint(0, 3)))
        for _ in range(randint(3, 5))
    ]


def role_ids(user):
    return tuple(sorted(role.id for role in user.roles))


def assert_db_state(users):
    """Verify that the DB user table contains the given user batch."""
    Session.expire_all()
    result = Session.execute(select(User)).scalars().all()
    assert set((row.id, row.name, row.email, row.active, row.verified, role_ids(row)) for row in result) \
           == set((user.id, user.name, user.email, user.active, user.verified, role_ids(user)) for user in users)


def assert_json_result(response, json, user):
    """Verify that the API result matches the given user object."""
    assert response.status_code == 200
    assert json['id'] == user.id
    assert json['name'] == user.name
    assert json['email'] == user.email
    assert json['active'] == user.active
    assert json['verified'] == user.verified
    assert tuple(sorted(json['role_ids'])) == role_ids(user)


def assert_json_results(response, json, users):
    """Verify that the API result list matches the given user batch."""
    items = json['items']
    assert json['total'] == len(items) == len(users)
    items.sort(key=lambda i: i['id'])
    users.sort(key=lambda u: u.id)
    for n, user in enumerate(users):
        assert_json_result(response, items[n], user)


@pytest.mark.parametrize('scopes', [
    [ODPScope.USER_READ],
    [],
    all_scopes,
    all_scopes_excluding(ODPScope.USER_READ),
])
def test_list_users(api, user_batch, scopes):
    authorized = ODPScope.USER_READ in scopes
    r = api(scopes).get('/user/')
    if authorized:
        assert_json_results(r, r.json(), user_batch)
    else:
        assert_forbidden(r)
    assert_db_state(user_batch)


@pytest.mark.parametrize('scopes', [
    [ODPScope.USER_READ],
    [],
    all_scopes,
    all_scopes_excluding(ODPScope.USER_READ),
])
def test_get_user(api, user_batch, scopes):
    authorized = ODPScope.USER_READ in scopes
    r = api(scopes).get(f'/user/{user_batch[2].id}')
    if authorized:
        assert_json_result(r, r.json(), user_batch[2])
    else:
        assert_forbidden(r)
    assert_db_state(user_batch)


def test_get_user_not_found(api, user_batch):
    scopes = [ODPScope.USER_READ]
    r = api(scopes).get('/user/foo')
    assert_not_found(r)
    assert_db_state(user_batch)


def test_create_user(api):
    r = api(all_scopes).post('/user/')
    assert_method_not_allowed(r)


@pytest.mark.parametrize('scopes', [
    [ODPScope.USER_ADMIN],
    [],
    all_scopes,
    all_scopes_excluding(ODPScope.USER_ADMIN),
])
def test_update_user(api, user_batch, scopes):
    authorized = ODPScope.USER_ADMIN in scopes
    modified_user_batch = user_batch.copy()
    modified_user_batch[2] = (user := UserFactory.build(
        id=user_batch[2].id,
        name=user_batch[2].name,
        email=user_batch[2].email,
        verified=user_batch[2].verified,
        roles=RoleFactory.create_batch(randint(0, 3)),
    ))
    r = api(scopes).put('/user/', json=dict(
        id=user.id,
        active=user.active,
        role_ids=role_ids(user),
    ))
    if authorized:
        assert_empty_result(r)
        assert_db_state(modified_user_batch)
    else:
        assert_forbidden(r)
        assert_db_state(user_batch)


def test_update_user_not_found(api, user_batch):
    scopes = [ODPScope.USER_ADMIN]
    user = UserFactory.build(
        id='foo',
        name=user_batch[2].name,
        email=user_batch[2].email,
        verified=user_batch[2].verified,
        roles=RoleFactory.create_batch(randint(0, 3)),
    )
    r = api(scopes).put('/user/', json=dict(
        id=user.id,
        active=user.active,
        role_ids=role_ids(user),
    ))
    assert_not_found(r)
    assert_db_state(user_batch)


@pytest.mark.parametrize('scopes', [
    [ODPScope.USER_ADMIN],
    [],
    all_scopes,
    all_scopes_excluding(ODPScope.USER_ADMIN),
])
def test_delete_user(api, user_batch, scopes):
    authorized = ODPScope.USER_ADMIN in scopes
    modified_user_batch = user_batch.copy()
    del modified_user_batch[2]
    r = api(scopes).delete(f'/user/{user_batch[2].id}')
    if authorized:
        assert_empty_result(r)
        assert_db_state(modified_user_batch)
    else:
        assert_forbidden(r)
        assert_db_state(user_batch)


def test_delete_user_not_found(api, user_batch):
    scopes = [ODPScope.USER_ADMIN]
    r = api(scopes).delete('/user/foo')
    assert_not_found(r)
    assert_db_state(user_batch)
