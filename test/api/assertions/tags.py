from datetime import datetime

from sqlalchemy import select

from odp.db.models import CollectionTag, CollectionTagAudit, PackageTag, PackageTagAudit, RecordTag, RecordTagAudit
from test import TestSession
from test.api.assertions import assert_new_timestamp

_tag_instance_classes = {
    'collection': CollectionTag,
    'package': PackageTag,
    'record': RecordTag,
}

_tag_instance_audit_classes = {
    'collection': CollectionTagAudit,
    'package': PackageTagAudit,
    'record': RecordTagAudit,
}


def assert_tag_instance_output(response, tag_instance, grant_type):
    """Assert that the API response matches the given tag instance."""
    assert response.status_code == 200
    json = response.json()
    user_id = tag_instance.get('user_id', 'odp.test.user')
    user_email = tag_instance.get('user_email', 'test@saeon.ac.za')
    assert json['tag_id'] == tag_instance['tag_id']
    assert json['user_id'] == (user_id if grant_type == 'authorization_code' else None)
    assert json['user_name'] == ('Test User' if grant_type == 'authorization_code' else None)
    assert json['user_email'] == (user_email if grant_type == 'authorization_code' else None)
    assert json['data'] == tag_instance['data']
    assert_new_timestamp(datetime.fromisoformat(json['timestamp']))
    assert json['cardinality'] == tag_instance['cardinality']
    assert json['public'] == tag_instance['public']
    assert json['vocabulary_id'] == tag_instance['vocabulary_id']
    assert json['keyword_id'] == tag_instance['keyword_id']
    assert json['keyword'] == tag_instance['keyword']


def assert_tag_instance_db_state(tag_type, grant_type, object_id, *tag_instances):
    """Assert that the relevant tag instance table data match the
    given array of tag instance objects/dicts.
    """
    tag_instance_cls = _tag_instance_classes[tag_type]

    result = TestSession.execute(select(tag_instance_cls)).scalars().all()
    result.sort(key=lambda r: r.timestamp)
    assert len(result) == len(tag_instances)

    for n, row in enumerate(result):
        assert getattr(row, f'{tag_type}_id') == object_id
        assert row.tag_type == tag_type
        assert_new_timestamp(row.timestamp)

        if isinstance(tag_instance := tag_instances[n], dict):
            user_id = tag_instance.get('user_id', 'odp.test.user')
            assert row.tag_id == tag_instance['tag_id']
            assert row.user_id == (user_id if grant_type == 'authorization_code' else None)
            assert row.vocabulary_id == tag_instance['vocabulary_id']
            assert row.keyword_id == tag_instance['keyword_id']
            assert row.data == tag_instance['data']
        else:
            assert row.tag_id == tag_instance.tag_id
            assert row.user_id == tag_instance.user_id
            assert row.vocabulary_id == tag_instance.vocabulary_id
            assert row.keyword_id == tag_instance.keyword_id
            assert row.data == tag_instance.data


def assert_tag_instance_audit_log(tag_type, grant_type, *entries):
    """Assert that the relevant tag instance audit table data match the
    given array of audit entries.

    Each entry is a dict with keys 'command', 'object_id' and 'tag_instance'.
    """
    tag_instance_audit_cls = _tag_instance_audit_classes[tag_type]

    result = TestSession.execute(select(tag_instance_audit_cls)).scalars().all()
    assert len(result) == len(entries)

    for n, row in enumerate(result):
        auth_client_id = entries[n]['tag_instance'].get('auth_client_id', 'odp.test.client')
        auth_user_id = entries[n]['tag_instance'].get('auth_user_id', 'odp.test.user' if grant_type == 'authorization_code' else None)

        assert row.client_id == auth_client_id
        assert row.user_id == auth_user_id
        assert row.command == entries[n]['command']
        assert_new_timestamp(row.timestamp)

        assert getattr(row, f'_{tag_type}_id') == entries[n]['object_id']
        assert row._tag_id == entries[n]['tag_instance']['tag_id']
        assert row._user_id == entries[n]['tag_instance'].get('user_id', auth_user_id)
        assert row._data == entries[n]['tag_instance']['data']
        assert row._keyword_id == entries[n]['tag_instance']['keyword_id']


def assert_tag_instance_audit_log_empty(tag_type):
    """Assert that the relevant tag instance audit table is empty."""
    tag_instance_audit_cls = _tag_instance_audit_classes[tag_type]
    assert TestSession.execute(select(tag_instance_audit_cls)).first() is None
