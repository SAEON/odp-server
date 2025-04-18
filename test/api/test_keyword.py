from random import choice, randint

import pytest
from sqlalchemy import select

from odp.const import ODPScope
from odp.db.models import Keyword, KeywordAudit
from test import TestSession
from test.api.assertions import assert_conflict, assert_forbidden, assert_new_timestamp, assert_not_found, assert_ok_null, assert_unprocessable
from test.factories import FactorySession, KeywordFactory, VocabularyFactory, create_keyword_data, create_keyword_key


@pytest.fixture
def keyword_batch(request):
    """Create and commit a batch of Keyword instances, which
    may include sub-keywords, recursively. Return a tuple of
    (top-level keywords, all keywords).
    """
    vocabs = VocabularyFactory.create_batch(3)
    keywords_top = []
    for n in range(randint(3, 5)):
        keywords_top += [KeywordFactory(vocabulary=choice(vocabs))]
    keywords_flat = FactorySession.execute(select(Keyword)).scalars().all()

    for keyword in keywords_flat:
        keyword.ids = [keyword.id]
        keyword.keys_ = [keyword.key]
        parent = keyword.parent
        while parent:
            keyword.ids.insert(0, parent.id)
            keyword.keys_.insert(0, parent.key)
            parent = parent.parent

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


def assert_json_result(response, json, keyword, hierarchy=False):
    """Assert that the API response matches the given keyword object."""
    assert response.status_code == 200
    assert json['vocabulary_id'] == keyword.vocabulary_id
    assert json['id'] == keyword.id
    assert json['key'] == keyword.key
    assert json['data'] == keyword.data
    assert json['status'] == keyword.status
    assert json['parent_id'] == keyword.parent_id
    if hierarchy:
        assert json['ids'] == keyword.ids
        assert json['keys_'] == keyword.keys_


def assert_json_results(response, json, keywords, hierarchy=False):
    """Assert that the API response list matches the given keyword batch."""
    items = json['items']
    assert json['total'] == len(items) == len(keywords)
    items.sort(key=lambda i: i['id'])
    keywords.sort(key=lambda k: k.id)
    for n, keyword in enumerate(keywords):
        assert_json_result(response, items[n], keyword, hierarchy)


@pytest.mark.require_scope(ODPScope.KEYWORD_READ_ALL)
def test_list_all_keywords(
        api,
        scopes,
        keyword_batch,
):
    authorized = ODPScope.KEYWORD_READ_ALL in scopes
    keywords_top, keywords_flat = keyword_batch

    r = api(scopes).get('/keyword/?size=0')

    if not authorized:
        assert_forbidden(r)
    else:
        assert_json_results(r, r.json(), keywords_flat, hierarchy=True)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_READ_ALL)
@pytest.mark.parametrize('error', [None, 'kw_404'])
def test_get_any_keyword(
        api,
        scopes,
        keyword_batch,
        error,
):
    authorized = ODPScope.KEYWORD_READ_ALL in scopes
    keywords_top, keywords_flat = keyword_batch

    old_ix = randint(0, len(keywords_flat) - 1)
    old_kw = keywords_flat[old_ix]
    kw_id = 0 if error == 'kw_404' else old_kw.id

    r = api(scopes).get(f'/keyword/{kw_id}')

    if not authorized:
        assert_forbidden(r)
    elif error == 'kw_404':
        assert_not_found(r)
    else:
        assert_json_result(r, r.json(), old_kw, hierarchy=True)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_READ)
@pytest.mark.parametrize('error', [None, 'vocab_404'])
@pytest.mark.parametrize('parent_key', [None, True])
@pytest.mark.parametrize('include_proposed', [False, True])
def test_list_keywords(
        api,
        scopes,
        keyword_batch,
        error,
        parent_key,
        include_proposed,
):
    authorized = ODPScope.KEYWORD_READ in scopes
    keywords_top, keywords_flat = keyword_batch

    vocab_id = 'foo' if error == 'vocab_404' else keywords_top[2].vocabulary_id
    statuses = ['approved']
    if include_proposed:
        statuses += ['proposed']

    if parent_key:
        candidate_children = list(filter(lambda k: k.vocabulary_id == vocab_id and k.parent_id is not None, keywords_flat))
        if not candidate_children:
            pytest.skip('empty candidate set')
        child_ix = randint(0, len(candidate_children) - 1)
        parent = candidate_children[child_ix].parent
        parent_arg = f'&parent_key={parent.key}'
        keywords_expected = list(filter(
            lambda k: k.vocabulary_id == vocab_id and k.status in statuses and k.parent_id == parent.id,
            keywords_flat
        ))
    else:
        parent_arg = ''
        keywords_expected = list(filter(
            lambda k: k.vocabulary_id == vocab_id and k.status in statuses,
            keywords_flat
        ))

    r = api(scopes).get(f'/keyword/{vocab_id}/?size=0{parent_arg}&include_proposed={include_proposed}')

    if not authorized:
        assert_forbidden(r)
    elif error == 'vocab_404':
        assert_not_found(r, 'Vocabulary not found')
    else:
        assert_json_results(r, r.json(), keywords_expected, hierarchy=True)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_READ)
@pytest.mark.parametrize('error', [None, 'unapproved_kw', 'wrong_vocab', 'unknown_kw'])
def test_get_keyword(
        api,
        scopes,
        keyword_batch,
        error,
):
    authorized = ODPScope.KEYWORD_READ in scopes
    keywords_top, keywords_flat = keyword_batch

    vocab_0_id = keywords_top[0].vocabulary_id
    if error == 'unapproved_kw':
        keywords = list(filter(lambda kw: kw.status != 'approved', keywords_flat))
    elif error == 'wrong_vocab':
        keywords = list(filter(lambda kw: kw.status == 'approved' and kw.vocabulary_id != vocab_0_id, keywords_flat))
    else:
        keywords = list(filter(lambda kw: kw.status == 'approved', keywords_flat))

    if not keywords:
        pytest.skip('filtered batch is empty')

    old_ix = randint(0, len(keywords) - 1)
    old_kw = keywords[old_ix]
    key = 'foo' if error == 'unknown_kw' else old_kw.key
    vocab_id = vocab_0_id if error == 'wrong_vocab' else old_kw.vocabulary_id

    r = api(scopes).get(f'/keyword/{vocab_id}/{key}')

    if not authorized:
        assert_forbidden(r)
    elif error is not None:
        assert_not_found(r)
    else:
        assert_json_result(r, r.json(), old_kw, hierarchy=True)

    assert_db_state(keywords_flat)
    assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_SUGGEST)
@pytest.mark.parametrize('is_child', [False, True])
@pytest.mark.parametrize('error', [None, 'vocab_404', 'parent_404', 'invalid_data', 'kw_conflict'])
def test_suggest_keyword(
        api,
        scopes,
        keyword_batch,
        is_child,
        error,
):
    _test_suggest_or_create_keyword(
        api,
        scopes,
        keyword_batch,
        is_child,
        error,
        ODPScope.KEYWORD_SUGGEST in scopes,
        'suggest',
    )


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
@pytest.mark.parametrize('is_child', [False, True])
@pytest.mark.parametrize('error', [None, 'vocab_404', 'parent_404', 'invalid_data', 'kw_conflict'])
def test_create_keyword(
        api,
        scopes,
        keyword_batch,
        is_child,
        error,
):
    _test_suggest_or_create_keyword(
        api,
        scopes,
        keyword_batch,
        is_child,
        error,
        ODPScope.KEYWORD_ADMIN in scopes,
        'create',
    )


def _test_suggest_or_create_keyword(
        api,
        scopes,
        keyword_batch,
        is_child,
        error,
        authorized,
        function,
):
    keywords_top, keywords_flat = keyword_batch

    new_kw_args = dict(
        vocabulary=(vocab := keywords_top[2].vocabulary),
        vocabulary_id=vocab.id,
    )

    if function == 'suggest':
        new_kw_args |= dict(
            status='proposed',
        )

    if error == 'kw_conflict':
        try:
            new_kw_args['key'] = choice(list(filter(
                lambda k: k.vocabulary_id == vocab.id and is_child == bool(k.parent_id),
                keywords_flat))).key
        except IndexError:
            pytest.skip('filtered list is empty; no child keys with which to conflict')

    if is_child:
        new_kw_args |= dict(
            parent=keywords_top[2],
            parent_id=keywords_top[2].id,
        )
        if error == 'parent_404':
            new_kw_args['parent_id'] = 0

    vocab_id = 'foo' if error == 'vocab_404' else vocab.id
    new_kw = keyword_build(**new_kw_args)
    if error == 'invalid_data':
        new_kw.data = create_keyword_data(new_kw, 0, True)

    api_func = api(scopes).post if function == 'suggest' else api(scopes).put
    api_args = dict(
        key=new_kw.key,
        data=new_kw.data,
        parent_id=new_kw.parent_id,
    )
    if function == 'create':
        api_args |= dict(
            status=new_kw.status,
        )

    r = api_func(f'/keyword/{vocab_id}/', json=api_args)

    changed = False
    if not authorized:
        assert_forbidden(r)
    elif error == 'vocab_404':
        assert_not_found(r, 'Vocabulary not found')
    elif error == 'parent_404' and is_child:
        assert_not_found(r, 'Parent keyword not found')
    elif error == 'invalid_data':
        assert_unprocessable(r, valid=False)
    elif error == 'kw_conflict':
        assert_conflict(r, f"Keyword '{new_kw.key}' already exists")
    else:
        new_kw.id = r.json()['id']
        assert_json_result(r, r.json(), new_kw)
        assert_db_state(keywords_flat + [new_kw])
        assert_audit_log(api.grant_type, dict(command='insert', keyword=new_kw))
        changed = True

    if not changed:
        assert_db_state(keywords_flat)
        assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
@pytest.mark.parametrize('change', [None, 'key', 'data', 'status', 'parent_id'])
@pytest.mark.parametrize('error', [None, 'kw_404', 'vocab_404', 'parent_404', 'invalid_data'])
def test_update_keyword(
        api,
        scopes,
        keyword_batch,
        change,
        error,
):
    authorized = ODPScope.KEYWORD_ADMIN in scopes
    keywords_top, keywords_flat = keyword_batch

    old_ix = randint(0, len(keywords_flat) - 1)
    old_kw = keywords_flat[old_ix]
    new_kw_args = dict(
        key=old_kw.key,
        data=old_kw.data.copy(),
        status=old_kw.status,
        parent=old_kw.parent,
        parent_id=old_kw.parent_id,
    )
    if change == 'key':
        new_kw_args['data']['key'] = new_kw_args['key'] = create_keyword_key(old_kw, -1)
    elif change == 'data':
        new_kw_args['data'] = create_keyword_data(old_kw, -1, invalid=error == 'invalid_data')
    elif change == 'status':
        new_kw_args['status'] = 'rejected' if old_kw.status == 'proposed' else 'proposed'
    elif change == 'parent_id':
        new_kw_args['parent'] = old_kw.parent.parent if old_kw.parent else None
        new_kw_args['parent_id'] = 0 if error == 'parent_404' else old_kw.parent.parent_id if old_kw.parent else None
        if new_kw_args['parent_id'] == old_kw.parent_id:
            change = None

    kw_id = 0 if error == 'kw_404' else old_kw.id
    vocab_id = 'foo' if error == 'vocab_404' else old_kw.vocabulary_id
    new_kw = keyword_build(
        id=old_kw.id,
        vocabulary=old_kw.vocabulary,
        vocabulary_id=old_kw.vocabulary_id,
        **new_kw_args
    )

    r = api(scopes).put(f'/keyword/{vocab_id}/{kw_id}', json=dict(
        key=new_kw.key,
        data=new_kw.data,
        status=new_kw.status,
        parent_id=new_kw.parent_id,
    ))

    changed = False
    if not authorized:
        assert_forbidden(r)
    elif error == 'kw_404':
        assert_not_found(r)
    elif error == 'vocab_404':
        assert_not_found(r, 'Vocabulary not found')
    elif error == 'parent_404' and change == 'parent_id':
        assert_not_found(r, 'Parent keyword not found')
    elif error == 'invalid_data' and change == 'data':
        assert_unprocessable(r, valid=False)
    elif change:
        assert_json_result(r, r.json(), new_kw)
        assert_db_state(keywords_flat[:old_ix] + [new_kw] + keywords_flat[old_ix + 1:])
        assert_audit_log(api.grant_type, dict(command='update', keyword=new_kw))
        changed = True
    else:
        assert_ok_null(r)

    if not changed:
        assert_db_state(keywords_flat)
        assert_no_audit_log()


@pytest.mark.require_scope(ODPScope.KEYWORD_ADMIN)
@pytest.mark.parametrize('error', [None, 'kw_404', 'vocab_404'])
def test_delete_keyword(
        api,
        scopes,
        keyword_batch,
        error,
):
    authorized = ODPScope.KEYWORD_ADMIN in scopes
    keywords_top, keywords_flat = keyword_batch

    old_ix = randint(0, len(keywords_flat) - 1)
    old_kw = keywords_flat[old_ix]
    deleted_kw = KeywordFactory.stub(
        vocabulary_id=old_kw.vocabulary_id,
        id=old_kw.id,
        key=old_kw.key,
        data=old_kw.data.copy(),
        status=old_kw.status,
        parent_id=old_kw.parent_id,
        has_children=True if old_kw.children else False,
    )
    kw_id = 0 if error == 'kw_404' else old_kw.id
    vocab_id = 'foo' if error == 'vocab_404' else old_kw.vocabulary_id

    r = api(scopes).delete(f'/keyword/{vocab_id}/{kw_id}')

    changed = False
    if not authorized:
        assert_forbidden(r)
    elif error in ('kw_404', 'vocab_404'):
        assert_not_found(r)
    elif deleted_kw.has_children:
        assert_unprocessable(r, f"Keyword '{deleted_kw.id}' has child keywords")
    else:
        assert_ok_null(r)
        assert_db_state(keywords_flat[:old_ix] + keywords_flat[old_ix + 1:])
        assert_audit_log(api.grant_type, dict(command='delete', keyword=deleted_kw))
        changed = True

    if not changed:
        assert_db_state(keywords_flat)
        assert_no_audit_log()
