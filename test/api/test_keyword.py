from random import choice, randint, sample

import pytest
from sqlalchemy import select

from odp.const import ODPScope
from odp.db.models import Keyword, KeywordAudit, Schema
from test import TestSession
from test.api import assert_conflict, assert_empty_result, assert_forbidden, assert_new_timestamp, assert_not_found, assert_unprocessable
from test.factories import FactorySession, KeywordFactory, create_keyword_data, create_keyword_key, fake


@pytest.fixture
def keyword_batch(request):
    """Create and commit a batch of Keyword instances, which
    may include sub-keywords, recursively. Return a tuple of
    (top-level keywords, all keywords).
    """
    keywords_top = KeywordFactory.create_batch(randint(3, 5))
    keywords_flat = FactorySession.execute(select(Keyword)).scalars().all()
    return keywords_top, keywords_flat


def keyword_build(**attr):
    """Build and return an uncommitted Keyword instance."""
    return KeywordFactory.build(
        children=[],
        **attr,
    )


def assert_db_state(keywords_flat):
    """Verify that the keyword table contains the given keyword batch."""
    result = TestSession.execute(select(Keyword)).scalars().all()
    result.sort(key=lambda k: k.id)
    keywords_flat.sort(key=lambda k: k.id)
    assert len(result) == len(keywords_flat)
    for n, row in enumerate(result):
        kw = keywords_flat[n]
        assert row.vocabulary_id == kw.vocabulary_id
        assert row.id == kw.id
        assert row.key == kw.key
        assert row.data == kw.data
        assert row.status == kw.status
        assert row.parent_id == kw.parent_id


def assert_audit_log(grant_type, *entries):
    result = TestSession.execute(select(KeywordAudit)).scalars().all()
    assert len(result) == len(entries)
    for n, row in enumerate(result):
        assert row.client_id == 'odp.test.client'
        assert row.user_id == ('odp.test.user' if grant_type == 'authorization_code' else None)
        assert row.command == entries[n]['command']
        assert_new_timestamp(row.timestamp)
        keyword = entries[n]['keyword']
        assert row._vocabulary_id == keyword.vocabulary_id
        assert row._id == keyword.id
        assert row._key == keyword.key
        assert row._data == keyword.data
        assert row._status == keyword.status
        assert row._parent_id == keyword.parent_id


def assert_no_audit_log():
    assert TestSession.execute(select(KeywordAudit)).first() is None


def assert_json_result(response, json, keyword, recurse=None):
    """Verify that the API result matches the given keyword object."""
    assert response.status_code == 200
    assert json['vocabulary_id'] == keyword.vocabulary_id
    assert json['id'] == keyword.id
    assert json['key'] == keyword.key
    assert json['data'] == keyword.data
    assert json['status'] == keyword.status
    assert json['parent_id'] == keyword.parent_id
    assert json['parent_key'] == (keyword.parent.key if keyword.parent_id else None)
    assert json['schema_id'] == keyword.vocabulary.schema_id

    if recurse:
        kw_children = list(filter(lambda k: k.status == 'approved', keyword.children)) \
            if recurse == 'approved' \
            else keyword.children
        assert len(json['child_keywords']) == len(kw_children)
        for i, kw_child in enumerate(kw_children):
            assert_json_result(response, json['child_keywords'][i], kw_child, recurse)
    else:
        assert 'child_keywords' not in json


def assert_json_results(response, json, keywords, recurse=False):
    """Verify that the API result list matches the given keyword batch."""
    items = json['items']
    assert json['total'] == len(items) == len(keywords)
    items.sort(key=lambda i: i['id'])
    keywords.sort(key=lambda k: k.id)
    for n, keyword in enumerate(keywords):
        assert_json_result(response, items[n], keyword, recurse)


@pytest.mark.require_scope(ODPScope.KEYWORD_READ_ALL)
def test_list_all_keywords(
        api,
        scopes,
        keyword_batch,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_READ_ALL in scopes

    r = api(scopes).get('/keyword/?size=0')
    if authorized:
        assert_json_results(r, r.json(), keywords_flat)
    else:
        assert_forbidden(r)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_READ_ALL)
@pytest.mark.parametrize('recurse', [False, True])
def test_get_any_keyword(
        api,
        scopes,
        keyword_batch,
        recurse,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_READ_ALL in scopes

    r = api(scopes).get(f'/keyword/{keywords_top[2].id}?recurse={recurse}')
    if authorized:
        assert_json_result(r, r.json(), keywords_top[2], 'all' if recurse else None)
    else:
        assert_forbidden(r)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_READ_ALL)
@pytest.mark.parametrize('recurse', [False, True])
def test_get_any_keyword_not_found(
        api,
        scopes,
        keyword_batch,
        recurse,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_READ_ALL in scopes

    r = api(scopes).get(f'/keyword/0?recurse={recurse}')
    if authorized:
        assert_not_found(r)
    else:
        assert_forbidden(r)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_READ)
def test_list_keywords(
        api,
        scopes,
        keyword_batch,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_READ in scopes

    vocabulary_id = keywords_top[2].vocabulary_id
    # TODO: ideally we should expect a 'chain of approval' starting at
    #  top-level keywords; see comments in the list_keywords API function.
    keywords_expected = list(filter(
        lambda k: k.vocabulary_id == vocabulary_id and k.status == 'approved',
        keywords_flat
    ))

    r = api(scopes).get(f'/keyword/{vocabulary_id}/?size=0')
    if authorized:
        assert_json_results(r, r.json(), keywords_expected)
    else:
        assert_forbidden(r)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_READ)
@pytest.mark.parametrize('recurse', [False, True])
def test_get_keyword(
        api,
        scopes,
        keyword_batch,
        recurse,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_READ in scopes
    client = api(scopes)

    for kw in sample(list(filter(lambda k: k.status == 'approved', keywords_flat)), 3):
        r = client.get(f'/keyword/{kw.vocabulary_id}/{kw.key}?recurse={recurse}')
        if authorized:
            assert_json_result(r, r.json(), kw, recurse='approved' if recurse else None)
        else:
            assert_forbidden(r)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_READ)
@pytest.mark.parametrize('recurse', [False, True])
def test_get_keyword_not_found(
        api,
        scopes,
        keyword_batch,
        recurse,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_READ in scopes
    client = api(scopes)

    # unapproved keyword => not found
    pop = list(filter(lambda kw: kw.status != 'approved', keywords_flat))
    k = min(4, len(pop))
    for kw in sample(pop, k):
        r = client.get(f'/keyword/{kw.vocabulary_id}/{kw.key}?recurse={recurse}')
        if authorized:
            assert_not_found(r)
        else:
            assert_forbidden(r)

    # approved but wrong vocab => not found
    vocab_0_id = keywords_top[0].vocabulary_id
    pop = list(filter(lambda kw: kw.status == 'approved' and kw.vocabulary_id != vocab_0_id, keywords_flat))
    k = min(4, len(pop))
    for kw in sample(pop, k):
        r = client.get(f'/keyword/{vocab_0_id}/{kw.key}?recurse={recurse}')
        if authorized:
            assert_not_found(r)
        else:
            assert_forbidden(r)

    # unknown keyword => not found
    r = client.get(f'/keyword/{vocab_0_id}/foo?recurse={recurse}')
    if authorized:
        assert_not_found(r)
    else:
        assert_forbidden(r)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_SUGGEST)
def test_suggest_keyword(
        api,
        scopes,
        keyword_batch,
):
    _test_create_keyword(
        api,
        'post',
        scopes,
        keyword_batch,
        ODPScope.KEYWORD_SUGGEST in scopes,
    )


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
def test_create_keyword(
        api,
        scopes,
        keyword_batch,
):
    _test_create_keyword(
        api,
        'put',
        scopes,
        keyword_batch,
        ODPScope.KEYWORD_ADMIN in scopes,
    )


def _test_create_keyword(
        api,
        method,
        scopes,
        keyword_batch,
        authorized,
):
    keywords_top, keywords_flat = keyword_batch
    client = api(scopes)
    api_func = client.post if method == 'post' else client.put

    vocab_1 = keywords_top[1].vocabulary
    kw_1_args = dict(
        data={'abbr': fake.word()},
        vocabulary=vocab_1,
        vocabulary_id=vocab_1.id,
    )
    if method == 'post':
        kw_1_args |= dict(status='proposed')

    keyword_1 = keyword_build(**kw_1_args)
    api_args = dict(
        key=keyword_1.key,
        data=keyword_1.data,
    )
    if method == 'put':
        api_args |= dict(status=keyword_1.status)

    r = api_func(f'/keyword/{vocab_1.id}/', json=api_args)

    if authorized:
        keyword_1.id = r.json().get('id')
        keyword_1.data['key'] = keyword_1.key
        assert_json_result(r, r.json(), keyword_1)

        kw_2_args = dict(
            data={'abbr': fake.word()},
            vocabulary=vocab_1,
            vocabulary_id=vocab_1.id,
            parent=keyword_1,
            parent_id=keyword_1.id,
        )
        if method == 'post':
            kw_2_args |= dict(status='proposed')

        keyword_2 = keyword_build(**kw_2_args)
        api_args = dict(
            key=keyword_2.key,
            data=keyword_2.data,
            parent_id=keyword_2.parent_id,
        )
        if method == 'put':
            api_args |= dict(status=keyword_2.status)

        r = api_func(f'/keyword/{vocab_1.id}/', json=api_args)

        keyword_2.id = r.json().get('id')
        keyword_2.data['key'] = keyword_2.key
        assert_json_result(r, r.json(), keyword_2)

        assert_db_state(keywords_flat + [keyword_1, keyword_2])
        assert_audit_log(
            api.grant_type,
            dict(command='insert', keyword=keyword_1),
            dict(command='insert', keyword=keyword_2),
        )

    else:
        assert_forbidden(r)
        assert_db_state(keywords_flat)
        assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_SUGGEST)
def test_suggest_keyword_parent_not_found(
        api,
        scopes,
        keyword_batch,
):
    _test_create_keyword_parent_not_found(
        api,
        'post',
        scopes,
        keyword_batch,
        ODPScope.KEYWORD_SUGGEST in scopes,
    )


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
def test_create_keyword_parent_not_found(
        api,
        scopes,
        keyword_batch,
):
    _test_create_keyword_parent_not_found(
        api,
        'put',
        scopes,
        keyword_batch,
        ODPScope.KEYWORD_ADMIN in scopes,
    )


def _test_create_keyword_parent_not_found(
        api,
        method,
        scopes,
        keyword_batch,
        authorized,
):
    keywords_top, keywords_flat = keyword_batch
    client = api(scopes)
    api_func = client.post if method == 'post' else client.put

    # unknown parent id
    api_args = dict(
        key=fake.word(),
        data={'abbr': fake.word()},
        parent_id=0,
    )
    if method == 'put':
        api_args |= dict(status='approved')

    r = api_func(f'/keyword/{keywords_top[1].vocabulary_id}/', json=api_args)
    if authorized:
        assert_not_found(r, 'Parent keyword not found')
    else:
        assert_forbidden(r)

    # parent in different vocab
    api_args = dict(
        key=fake.word(),
        data={'abbr': fake.word()},
        parent_id=keywords_top[2].id,
    )
    if method == 'put':
        api_args |= dict(status='approved')

    r = api_func(f'/keyword/{keywords_top[1].vocabulary_id}/', json=api_args)
    if authorized:
        assert_not_found(r, 'Parent keyword not found')
    else:
        assert_forbidden(r)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_SUGGEST)
def test_suggest_keyword_conflict(
        api,
        scopes,
        keyword_batch,
):
    _test_create_keyword_conflict(
        api,
        'post',
        scopes,
        keyword_batch,
        ODPScope.KEYWORD_SUGGEST in scopes,
    )


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
def test_create_keyword_conflict(
        api,
        scopes,
        keyword_batch,
):
    _test_create_keyword_conflict(
        api,
        'put',
        scopes,
        keyword_batch,
        ODPScope.KEYWORD_ADMIN in scopes,
    )


def _test_create_keyword_conflict(
        api,
        method,
        scopes,
        keyword_batch,
        authorized,
):
    keywords_top, keywords_flat = keyword_batch
    client = api(scopes)
    api_func = client.post if method == 'post' else client.put

    for kw in sample(keywords_flat, 3):
        api_args = dict(
            key=kw.key,
            data={'abbr': fake.word()},
            parent_id=kw.parent.parent_id if kw.parent and randint(0, 1) else kw.parent_id,
        )
        if method == 'put':
            api_args |= dict(status=choice(['proposed', 'approved', 'rejected', 'obsolete']))

        r = api_func(f'/keyword/{kw.vocabulary_id}/', json=api_args)
        if authorized:
            assert_conflict(r, f"Keyword '{kw.key}' already exists")
        else:
            assert_forbidden(r)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_SUGGEST)
def test_suggest_keyword_invalid_data(
        api,
        scopes,
        keyword_batch,
):
    _test_create_keyword_invalid_data(
        api,
        'post',
        scopes,
        keyword_batch,
        ODPScope.KEYWORD_SUGGEST in scopes,
    )


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
def test_create_keyword_invalid_data(
        api,
        scopes,
        keyword_batch,
):
    _test_create_keyword_invalid_data(
        api,
        'put',
        scopes,
        keyword_batch,
        ODPScope.KEYWORD_ADMIN in scopes,
    )


def _test_create_keyword_invalid_data(
        api,
        method,
        scopes,
        keyword_batch,
        authorized,
):
    keywords_top, keywords_flat = keyword_batch
    client = api(scopes)
    api_func = client.post if method == 'post' else client.put

    api_args = dict(
        key=fake.word(),
        data={'additional_prop': 'foo'},
    )
    if method == 'put':
        api_args |= dict(status='approved')

    r = api_func(f'/keyword/{keywords_top[1].vocabulary_id}/', json=api_args)
    if authorized:
        assert_unprocessable(r, valid=False)
    else:
        assert_forbidden(r)

    api_args = dict(
        key=fake.word(),
        data={'ror': 'invalid ror'},
        parent_id=keywords_top[1].id,
    )
    if method == 'put':
        api_args |= dict(status='approved')

    r = api_func(f'/keyword/{keywords_top[1].vocabulary_id}/', json=api_args)
    if authorized:
        assert_unprocessable(r, valid=False)
    else:
        assert_forbidden(r)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
@pytest.mark.parametrize('change', ['key', 'data', 'status', 'parent_id', None])
def test_update_keyword(
        api,
        scopes,
        keyword_batch,
        change,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_ADMIN in scopes

    old_kw = keywords_flat[-1]
    new_kw_args = dict(
        key=old_kw.key,
        data=old_kw.data.copy(),
        status=old_kw.status,
        parent=old_kw.parent,
        parent_id=old_kw.parent_id,
    )
    changed = True
    if change == 'key':
        new_kw_args['data']['key'] = new_kw_args['key'] = create_keyword_key(old_kw, -1)
    elif change == 'data':
        new_kw_args['data'] = create_keyword_data(old_kw, -1)
    elif change == 'status':
        new_kw_args['status'] = 'rejected' if old_kw.status == 'proposed' else 'proposed'
    elif change == 'parent_id':
        new_kw_args['parent'] = old_kw.parent.parent if old_kw.parent else None
        new_kw_args['parent_id'] = old_kw.parent.parent_id if old_kw.parent else None
        changed = new_kw_args['parent_id'] != old_kw.parent_id
    else:
        changed = False

    new_kw = keyword_build(
        id=old_kw.id,
        vocabulary=old_kw.vocabulary,
        vocabulary_id=old_kw.vocabulary_id,
        **new_kw_args
    )

    r = api(scopes).put(f'/keyword/{old_kw.vocabulary_id}/{old_kw.id}', json=dict(
        key=new_kw.key,
        data=new_kw.data,
        status=new_kw.status,
        parent_id=new_kw.parent_id,
    ))

    if authorized:
        assert_json_result(r, r.json(), new_kw)
        if changed:
            assert_db_state(keywords_flat[:-1] + [new_kw])
            assert_audit_log(api.grant_type, dict(command='update', keyword=new_kw))
        else:
            assert_db_state(keywords_flat)
            assert_no_audit_log()
    else:
        assert_forbidden(r)
        assert_db_state(keywords_flat)
        assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
def test_set_keyword_old(
        api,
        scopes,
        keyword_batch,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_ADMIN in scopes
    client = api(scopes)

    keywords_flat.reverse()
    audit = []
    for n in range(4):
        create = n == 0 or randint(0, 1)
        for i, kw in enumerate(keywords_flat):
            if kw.id.count('/') == n:
                data = {'abbr': fake.word(), 'title': fake.company()}
                child_schema = FactorySession.execute(select(Schema)).scalars().first() if randint(0, 1) else None

                if create:
                    kw_in = keyword_build(
                        parent_id=kw.id,
                        data=data,
                        child_schema=child_schema,
                    )
                    audit += [dict(command='insert', keyword=kw_in)]
                else:
                    kw_in = keyword_build(
                        parent_id=kw.parent_id,
                        id=kw.id,
                        data=data,
                        child_schema=child_schema,
                    )
                    audit += [dict(command='update', keyword=kw_in, replace=i)]

                r = client.put(f'/keyword/{kw_in.id}', json=dict(
                    parent_id=kw_in.parent_id,
                    data=kw_in.data,
                    status=kw_in.status,
                    child_schema_id=kw_in.child_schema_id,
                ))

                if authorized:
                    assert_json_result(r, r.json(), kw_in)
                else:
                    assert_forbidden(r)
                break

    if authorized:
        replace_indices = [entry.pop('replace', None) for entry in audit]
        for replace_index in reversed(sorted(filter(lambda i: i is not None, replace_indices))):
            keywords_flat.pop(replace_index)
        assert_db_state(keywords_flat + [entry.get('keyword') for entry in audit])
        assert_audit_log(api.grant_type, *audit)

    else:
        assert_db_state(keywords_flat)
        assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
def test_set_keyword_parent_not_found(
        api,
        scopes,
        keyword_batch,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_ADMIN in scopes
    client = api(scopes)

    r = client.put('/keyword/foo/bar', json=dict(
        data={'abbr': fake.word(), 'title': fake.company()},
        parent_id='foo',
    ))
    if authorized:
        assert_not_found(r, "Parent keyword 'foo' does not exist")
    else:
        assert_forbidden(r)

    r = client.put(f'/keyword/{(parent_id := keywords_top[2].id)}/foo/bar', json=dict(
        data={'abbr': fake.word(), 'title': fake.company()},
        parent_id=f'{parent_id}/foo',
    ))
    if authorized:
        assert_not_found(r, f"Parent keyword '{parent_id}/foo' does not exist")
    else:
        assert_forbidden(r)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
@pytest.mark.parametrize('create', [False, True])
def test_set_keyword_invalid_parent(
        api,
        scopes,
        keyword_batch,
        create,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_ADMIN in scopes
    client = api(scopes)

    # top-level keyword (vocab) cannot be created/updated via API
    kw_id = 'foo' if create else keywords_top[0].id
    r = client.put(f'/keyword/{kw_id}', json=dict(
        data={'abbr': fake.word(), 'title': fake.company()},
        parent_id='',
    ))
    if authorized:
        assert_unprocessable(r)
    else:
        assert_forbidden(r)

    r = client.put(f'/keyword/{kw_id}/bar', json=dict(
        data={'abbr': fake.word(), 'title': fake.company()},
        parent_id='bar',
    ))
    if authorized:
        assert_unprocessable(r, f"'bar' cannot be a parent of '{kw_id}/bar'")
    else:
        assert_forbidden(r)

    # self or sibling cannot be parent
    for n in range(1, 4):
        for kw in reversed(keywords_flat):
            if kw.id.count('/') == n:
                kw_id = f'{kw.parent.id}/foo' if create else kw.id
                r = client.put(f'/keyword/{kw_id}', json=dict(
                    data={'abbr': fake.word(), 'title': fake.company()},
                    parent_id=f'{kw.id}',
                ))
                if authorized:
                    assert_unprocessable(r, f"'{kw.id}' cannot be a parent of '{kw_id}'")
                else:
                    assert_forbidden(r)
                break

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
def test_set_keyword_invalid_data(
        api,
        scopes,
        keyword_batch,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_ADMIN in scopes

    r = api(scopes).put(f'/keyword/{keywords_top[2].id}/foo', json=dict(
        data={'abbr': fake.word()},  # missing required property 'title'
        parent_id=keywords_top[2].id,
    ))
    if authorized:
        assert_unprocessable(r, valid=False)
    else:
        assert_forbidden(r)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
def test_delete_keyword(
        api,
        scopes,
        keyword_batch,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_ADMIN in scopes
    client = api(scopes)

    keywords_flat.reverse()
    audit = []
    for n in range(4):
        for i, kw in enumerate(keywords_flat):
            if kw.id.count('/') == n:
                if can_delete := not kw.children:
                    audit += [dict(command='delete', keyword=kw, index=i)]

                r = client.delete(f'/keyword/{kw.id}')

                if authorized:
                    if can_delete:
                        assert_empty_result(r)
                    else:
                        assert_unprocessable(r, f"Keyword '{kw.id}' cannot be deleted as it has sub-keywords")
                else:
                    assert_forbidden(r)
                break

    if authorized:
        delete_indices = [entry.pop('index') for entry in audit]
        for delete_index in reversed(sorted(delete_indices)):
            keywords_flat.pop(delete_index)
        assert_db_state(keywords_flat)
        assert_audit_log(api.grant_type, *audit)
    else:
        assert_db_state(keywords_flat)
        assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
def test_delete_keyword_not_found(
        api,
        scopes,
        keyword_batch,
):
    keywords_top, keywords_flat = keyword_batch
    authorized = ODPScope.KEYWORD_ADMIN in scopes
    client = api(scopes)

    for n in range(2):
        for kw in reversed(keywords_flat):
            if kw.id.count('/') == n:
                r = client.delete(f'/keyword/{kw.id}/foo')
                if authorized:
                    assert_not_found(r, f"Keyword '{kw.id}/foo' does not exist")
                else:
                    assert_forbidden(r)
                break

    assert_db_state(keywords_flat)
    assert_no_audit_log()
