from random import randint

import pytest
from sqlalchemy import select

from odp.const import ODPScope
from odp.const.db import SubmissionStatus
from odp.db import Session
from odp.db.models import Submission
from test.api import assert_empty_result, assert_forbidden
from test.factories import SubmissionFactory


@pytest.fixture
def submission_batch():
    return [SubmissionFactory() for _ in range(randint(3, 5))]


def assert_db_submission_state(submissions):
    Session.expire_all()
    result = Session.execute(select(Submission)).scalars().all()
    result.sort(key=lambda s: s.id)
    submissions.sort(key=lambda s: s.id)

    assert len(result) == len(submissions)
    for n, row in enumerate(result):
        assert row.id == submissions[n].id
        assert row.user_id == submissions[n].user_id
        assert row.status == submissions[n].status
        assert row.data == submissions[n].data
        assert row.dataset_file_name == submissions[n].dataset_file_name
        assert row.doi == submissions[n].doi
        assert row.schema_id == submissions[n].schema_id


def assert_json_submission_result(json_data, submission):
    assert json_data['id'] == submission.id
    assert json_data['title'] == submission.data.get('title')
    assert json_data['status'] == submission.status.value


@pytest.mark.require_scope(ODPScope.SUBMISSION_READ)
def test_list_user_submissions(api, submission_batch, scopes):
    authorized = ODPScope.SUBMISSION_READ in scopes
    target_submission = submission_batch[0]

    r = api(scopes).get('/submission/user_submissions', params={'user_id': target_submission.user_id})

    if authorized:
        assert r.status_code == 200
        json_items = r.json()['items']

        expected_items = [s for s in submission_batch if s.user_id == target_submission.user_id]
        assert len(json_items) == len(expected_items)
        assert_json_submission_result(json_items[0], expected_items[0])
    else:
        assert_forbidden(r)

    assert_db_submission_state(submission_batch)


@pytest.mark.require_scope(ODPScope.SUBMISSION_READ)
def test_get_submission(api, submission_batch, scopes):
    authorized = ODPScope.SUBMISSION_READ in scopes
    target = submission_batch[0]

    r = api(scopes).get(f'/submission/{target.id}', params={'user_id': target.user_id})

    if authorized:
        assert r.status_code == 200
        assert r.json()['id'] == target.id
        assert r.json()['data'] == target.data
    else:
        assert_forbidden(r)


@pytest.mark.require_scope(ODPScope.SUBMISSION_WRITE)
def test_create_submission(api, submission_batch, scopes):
    authorized = ODPScope.SUBMISSION_WRITE in scopes

    mock_submission = SubmissionFactory.build()

    r = api(scopes).post('/submission/', json=dict(
        user_id=mock_submission.user_id,
        data=mock_submission.data
    ))

    if authorized:
        assert r.status_code == 200
        assert r.json()['user_id'] == mock_submission.user_id
        assert r.json()['status'] == SubmissionStatus.in_progress.value

        Session.expire_all()
        db_record = Session.get(Submission, r.json()['id'])
        assert db_record is not None
        assert db_record.data == mock_submission.data
    else:
        assert_forbidden(r)


@pytest.mark.require_scope(ODPScope.SUBMISSION_WRITE)
def test_update_submission(api, submission_batch, scopes):
    authorized = ODPScope.SUBMISSION_WRITE in scopes
    target = submission_batch[0]

    updated_data = target.data.copy()
    updated_data['title'] = "An Updated Octopus Garden Title"

    r = api(scopes).put(
        f'/submission/{target.id}',
        params={'user_id': target.user_id},
        json=updated_data
    )

    if authorized:
        assert r.status_code == 200
        assert r.json()['data']['title'] == "An Updated Octopus Garden Title"
        Session.expire_all()
        assert Session.get(Submission, target.id).data['title'] == "An Updated Octopus Garden Title"
    else:
        assert_forbidden(r)


@pytest.mark.require_scope(ODPScope.SUBMISSION_WRITE)
def test_submit_submission(api, submission_batch, scopes):
    authorized = ODPScope.SUBMISSION_WRITE in scopes
    target = submission_batch[0]
    target.status = SubmissionStatus.in_progress
    target.save()

    r = api(scopes).post(f'/submission/submit/{target.id}', params={'user_id': target.user_id})

    if authorized:
        assert r.status_code == 200
        assert r.json()['status'] == SubmissionStatus.submitted.value
        Session.expire_all()
        assert Session.get(Submission, target.id).status == SubmissionStatus.submitted
    else:
        assert_forbidden(r)


@pytest.mark.require_scope(ODPScope.SUBMISSION_DELETE)
def test_delete_submission(api, submission_batch, scopes, monkeypatch):
    authorized = ODPScope.SUBMISSION_DELETE in scopes
    target = submission_batch[0]

    target_id = target.id
    target_user_id = target.user_id

    mock_delete = lambda *args, **kwargs: None
    monkeypatch.setattr('odp.api.routers.submission.delete_folder_from_nextcloud', mock_delete)

    r = api(scopes).delete(f'/submission/{target_id}', params={'user_id': target_user_id})

    if authorized:
        assert_empty_result(r)
        Session.expire_all()
        assert Session.get(Submission, target_id) is None
    else:
        assert_forbidden(r)


@pytest.mark.require_scope(ODPScope.SUBMISSION_ADMIN)
def test_admin_list_submissions(api, submission_batch, scopes):
    authorized = ODPScope.SUBMISSION_ADMIN in scopes

    r = api(scopes).get('/submission/', params={'status': 'all'})

    if authorized:
        assert r.status_code == 200
        assert r.json()['total'] == len(submission_batch)
    else:
        assert_forbidden(r)
