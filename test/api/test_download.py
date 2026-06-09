"""
Tests for the /download router.

Coverage:
- POST /audit  — creates a DownloadAudit record (no auth required)
- GET  /logs   — paginated log listing, requires CATALOG_READ scope
- GET  /stats  — download statistics, requires CATALOG_READ scope
- GET  /export/csv — CSV export, requires CATALOG_READ scope
"""
from random import randint

import pytest

from odp.const import ODPScope
from odp.db import Session
from odp.db.models import DownloadAudit
from test.api import assert_forbidden
from test.factories import DownloadAuditFactory


# Fixtures

@pytest.fixture
def download_batch():
    return [DownloadAuditFactory() for _ in range(randint(3, 5))]


# POST /audit

def test_create_audit_minimal(api):
    payload = {
        'client_id': 'test.client',
        'success': True,
    }
    r = api([]).post('/download/audit', json=payload)
    assert r.status_code == 201
    assert r.json()['status'] == 'ok'
    assert isinstance(r.json()['audit_id'], int)


def test_create_audit_full(api):
    payload = {
        'client_id': 'MIMS.Catalog.UI',
        'user_id': 'user-abc',
        'download_url': 'http://example.com/data.zip',
        'file_size': 204800,
        'success': True,
        'meta': {
            'name': 'Test User',
            'email': 'test@saeon.ac.za',
            'organisation': 'SAEON',
            'doi': '10.1234/test',
        },
    }
    r = api([]).post('/download/audit', json=payload)
    assert r.status_code == 201

    audit = Session.get(DownloadAudit, r.json()['audit_id'])
    assert audit is not None
    assert audit.client_id == 'MIMS.Catalog.UI'
    assert audit.file_size == 204800
    assert audit.meta['email'] == 'test@saeon.ac.za'


def test_create_audit_invalid_payload(api):
    r = api([]).post('/download/audit', json=[1, 2, 3])
    assert r.status_code == 400


# GET /logs

@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_get_logs(api, download_batch, scopes):
    authorized = ODPScope.CATALOG_READ in scopes
    r = api(scopes).get('/download/logs')
    if authorized:
        assert r.status_code == 200
        data = r.json()
        assert data['total'] == len(download_batch)
        assert len(data['items']) == len(download_batch)
        assert data['page'] == 1
    else:
        assert_forbidden(r)


@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_get_logs_pagination(api, scopes):
    [DownloadAuditFactory() for _ in range(10)]
    authorized = ODPScope.CATALOG_READ in scopes
    r = api(scopes).get('/download/logs?page=1&size=3')
    if authorized:
        assert r.status_code == 200
        data = r.json()
        assert len(data['items']) <= 3
        assert data['total'] == 10
    else:
        assert_forbidden(r)


@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_get_logs_empty(api, scopes):
    authorized = ODPScope.CATALOG_READ in scopes
    r = api(scopes).get('/download/logs')
    if authorized:
        assert r.status_code == 200
        assert r.json()['total'] == 0
        assert r.json()['items'] == []
    else:
        assert_forbidden(r)


# GET /stats

@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_get_stats(api, download_batch, scopes):
    authorized = ODPScope.CATALOG_READ in scopes
    r = api(scopes).get('/download/stats')
    if authorized:
        assert r.status_code == 200
        data = r.json()
        assert data['total_downloads'] == len(download_batch)
        assert 'unique_users' in data
        assert 'successful_downloads' in data
        assert 'failed_downloads' in data
    else:
        assert_forbidden(r)


@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_get_stats_empty(api, scopes):
    authorized = ODPScope.CATALOG_READ in scopes
    r = api(scopes).get('/download/stats')
    if authorized:
        assert r.status_code == 200
        assert r.json()['total_downloads'] == 0
    else:
        assert_forbidden(r)


@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_get_stats_invalid_date(api, scopes):
    authorized = ODPScope.CATALOG_READ in scopes
    r = api(scopes).get('/download/stats?start_date=not-a-date')
    if authorized:
        assert r.status_code == 400
    else:
        assert_forbidden(r)


# GET /export/csv

@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_export_csv(api, download_batch, scopes):
    authorized = ODPScope.CATALOG_READ in scopes
    r = api(scopes).get('/download/export/csv')
    if authorized:
        assert r.status_code == 200
        assert 'text/csv' in r.headers.get('content-type', '')
        lines = r.text.strip().splitlines()
        # header + one row per record
        assert len(lines) == len(download_batch) + 1
        assert lines[0].startswith('ID,Timestamp')
    else:
        assert_forbidden(r)


@pytest.mark.require_scope(ODPScope.CATALOG_READ)
def test_export_csv_empty(api, scopes):
    authorized = ODPScope.CATALOG_READ in scopes
    r = api(scopes).get('/download/export/csv')
    if authorized:
        assert r.status_code == 200
        lines = r.text.strip().splitlines()
        assert len(lines) == 1  # header only
    else:
        assert_forbidden(r)
