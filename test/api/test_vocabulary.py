from random import randint

import pytest
from sqlalchemy import select

from odp.const import ODPScope
from odp.db.models import Keyword, Vocabulary
from test import TestSession
from test.api.assertions import assert_forbidden, assert_not_found
from test.factories import KeywordFactory, VocabularyFactory


@pytest.fixture
def vocabulary_batch():
    """Create and commit a batch of Vocabulary instances,
    with associated keywords."""
    vocabs = VocabularyFactory.create_batch(randint(3, 5))
    for vocab in vocabs:
        KeywordFactory.create_batch(randint(0, 4), vocabulary=vocab)
    return vocabs


def assert_db_state(vocabularies):
    """Verify that the DB vocabulary table contains the given vocabulary batch."""
    result = TestSession.execute(select(Vocabulary)).scalars().all()
    result.sort(key=lambda v: v.id)
    vocabularies.sort(key=lambda v: v.id)
    assert len(result) == len(vocabularies)
    for n, row in enumerate(result):
        assert row.id == vocabularies[n].id
        assert row.uri == vocabularies[n].uri
        assert row.schema_id == vocabularies[n].schema_id
        assert row.schema_type == 'keyword'
        assert row.static == vocabularies[n].static


def assert_json_result(response, json, vocabulary):
    """Verify that the API result matches the given vocabulary object."""
    assert response.status_code == 200
    assert json['id'] == vocabulary.id
    assert json['uri'] == vocabulary.uri
    assert json['schema_id'] == vocabulary.schema_id
    assert json['schema_uri'] == vocabulary.schema.uri
    assert json['schema_']['$id'] == vocabulary.schema.uri
    assert json['static'] == vocabulary.static

    db_keywords = TestSession.execute(
        select(Keyword).where(Keyword.vocabulary_id == vocabulary.id)
    ).scalars().all()
    assert json['keyword_count'] == len(db_keywords)


def assert_json_results(response, json, vocabularies):
    """Verify that the API result list matches the given vocabulary batch."""
    items = json['items']
    assert json['total'] == len(items) == len(vocabularies)
    items.sort(key=lambda i: i['id'])
    vocabularies.sort(key=lambda v: v.id)
    for n, vocabulary in enumerate(vocabularies):
        assert_json_result(response, items[n], vocabulary)


@pytest.mark.require_scope(ODPScope.VOCABULARY_READ)
def test_list_vocabularies(
        api,
        vocabulary_batch,
        scopes,
):
    authorized = ODPScope.VOCABULARY_READ in scopes

    r = api(scopes).get('/vocabulary/')

    if not authorized:
        assert_forbidden(r)
    else:
        assert_json_results(r, r.json(), vocabulary_batch)

    assert_db_state(vocabulary_batch)


@pytest.mark.require_scope(ODPScope.VOCABULARY_READ)
@pytest.mark.parametrize('error', [None, '404'])
def test_get_vocabulary(
        api,
        vocabulary_batch,
        scopes,
        error,
):
    authorized = ODPScope.VOCABULARY_READ in scopes
    vocab_id = 'foo' if error == '404' else vocabulary_batch[2].id

    r = api(scopes).get(f'/vocabulary/{vocab_id}')

    if not authorized:
        assert_forbidden(r)
    elif error == '404':
        assert_not_found(r)
    else:
        assert_json_result(r, r.json(), vocabulary_batch[2])

    assert_db_state(vocabulary_batch)
