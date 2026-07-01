"""
Tests for record_dataset_bundler.

Coverage:
- create_safe_folder_name: sanitisation and truncation
- _ensure_extension: extension inference from URL and Content-Type
- bundle_catalog_records: input validation, success path, missing records,
  missing metadata, data file included in ZIP
"""
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from odp.lib.record_dataset_bundler import (
    _ensure_extension,
    bundle_catalog_records,
    create_safe_folder_name,
)

USER_DATA = {'name': 'Jane Doe', 'email': 'jane@saeon.ac.za', 'organisation': 'SAEON'}

SAMPLE_METADATA = {
    'titles': [{'title': 'Ocean Temperature Dataset'}],
    'doi': '10.1234/ocean',
    'publisher': 'SAEON',
    'publicationYear': 2024,
    'creators': [{'name': 'Smith, J', 'affiliation': [], 'nameIdentifiers': []}],
    'descriptions': [{'descriptionType': 'Abstract', 'description': 'Test abstract.'}],
    'contributors': [],
    'rightsList': [],
    'geoLocations': [],
    'subjects': [],
    'dates': [{'dateType': 'Valid', 'date': '2020-01-01/2023-12-31'}],
}


# ============================================================================
# create_safe_folder_name
# ============================================================================

def test_safe_folder_name_basic():
    assert create_safe_folder_name('10.1234/ocean', 'Ocean Data 2024') == '10.1234_ocean_Ocean_Data_2024'


def test_safe_folder_name_strips_invalid_chars():
    result = create_safe_folder_name('10.1234/ocean', 'Dataset: 2024/01 <Test>')
    for char in '<>:"/\\|?*':
        assert char not in result


def test_safe_folder_name_collapses_underscores():
    result = create_safe_folder_name('10.1234/ocean', 'A   B')
    assert '__' not in result


def test_safe_folder_name_truncates():
    result = create_safe_folder_name('10.1234/ocean', 'A' * 300)
    assert len(result) <= 200


def test_safe_folder_name_truncation_preserves_doi():
    doi = '10.1234/ocean'
    result = create_safe_folder_name(doi, 'A' * 300)
    assert result.startswith('10.1234_ocean_')


def test_safe_folder_name_empty_returns_untitled():
    assert create_safe_folder_name('', '') == 'Untitled'


def test_safe_folder_name_none_returns_untitled():
    assert create_safe_folder_name(None, None) == 'Untitled'


def test_safe_folder_name_doi_only():
    assert create_safe_folder_name('10.1234/ocean', '') == '10.1234_ocean'


def test_safe_folder_name_title_only():
    assert create_safe_folder_name('', 'Ocean Data 2024') == 'Ocean_Data_2024'


# ============================================================================
# _ensure_extension
# ============================================================================

def test_ensure_extension_preserves_existing():
    assert _ensure_extension('data.nc', 'http://example.com/data.nc', None) == 'data.nc'


def test_ensure_extension_infers_from_url():
    result = _ensure_extension('data', 'http://example.com/file.csv', None)
    assert result == 'data.csv'


def test_ensure_extension_infers_from_content_type():
    result = _ensure_extension('data', 'http://example.com/file', 'text/plain')
    assert result.startswith('data')
    assert result != 'data'


def test_ensure_extension_no_info_available():
    assert _ensure_extension('data', 'http://example.com/file', None) == 'data'


# ============================================================================
# Input validation
# ============================================================================

def test_bundle_empty_record_ids_raises():
    with pytest.raises(ValueError, match='record_ids cannot be empty'):
        bundle_catalog_records([], USER_DATA)


def test_bundle_missing_required_user_fields_raises():
    with pytest.raises(ValueError, match='user_data missing required fields'):
        bundle_catalog_records(['rec-1'], {'name': 'A'})


def test_bundle_blank_user_fields_raises():
    with pytest.raises(ValueError, match='user_data missing required fields'):
        bundle_catalog_records(['rec-1'], {'name': '', 'email': '', 'organisation': ''})


# ============================================================================
# bundle_catalog_records — mocked DB and services
# ============================================================================

def _mock_catalog_record(record_id, doi, metadata):
    rec = MagicMock()
    rec.record_id = record_id
    rec.record.doi = doi
    rec.published_record = {'metadata_records': [{'metadata': metadata}]}
    return rec


def _mock_session(catalog_records):
    mock = MagicMock()
    mock.__enter__.return_value.execute.return_value.scalars.return_value.all.return_value = catalog_records
    return mock


@patch('odp.lib.record_dataset_bundler.fetch_external_file', return_value=(None, None))
@patch('odp.lib.record_dataset_bundler.generate_pdf')
@patch('odp.lib.record_dataset_bundler.Session')
def test_bundle_single_record_success(mock_session, mock_pdf, mock_fetch):
    mock_session.return_value = _mock_session([_mock_catalog_record('rec-1', '10.1234/ocean', SAMPLE_METADATA)])
    mock_pdf.return_value = BytesIO(b'%PDF-fake')

    result = bundle_catalog_records(['rec-1'], USER_DATA)

    assert result.record_count == 1
    assert result.failed_count == 0
    assert result.total_size > 0
    assert os.path.exists(result.zip_path)
    os.unlink(result.zip_path)


@patch('odp.lib.record_dataset_bundler.Session')
def test_bundle_record_not_found(mock_session):
    mock_session.return_value = _mock_session([])

    result = bundle_catalog_records(['missing-rec'], USER_DATA)

    assert result.record_count == 0
    assert result.failed_count == 1
    assert result.failed[0]['reason'] == 'not_found_or_not_published'
    os.unlink(result.zip_path)


@patch('odp.lib.record_dataset_bundler.fetch_external_file')
@patch('odp.lib.record_dataset_bundler.generate_pdf')
@patch('odp.lib.record_dataset_bundler.Session')
def test_bundle_includes_data_file(mock_session, mock_pdf, mock_fetch):
    metadata = {
        **SAMPLE_METADATA,
        'immutableResource': {
            'resourceDownload': {'downloadURL': 'http://example.com/data.nc', 'fileName': 'ocean.nc'}
        },
    }
    mock_session.return_value = _mock_session([_mock_catalog_record('rec-1', '10.1234/ocean', metadata)])
    mock_pdf.return_value = BytesIO(b'%PDF-fake')
    mock_fetch.return_value = (b'netcdf_bytes', 'application/x-netcdf')

    result = bundle_catalog_records(['rec-1'], USER_DATA)

    assert result.record_count == 1
    assert result.total_size > len(b'%PDF-fake')
    os.unlink(result.zip_path)


@patch('odp.lib.record_dataset_bundler.fetch_external_file', return_value=(None, None))
@patch('odp.lib.record_dataset_bundler.generate_pdf')
@patch('odp.lib.record_dataset_bundler.Session')
def test_bundle_missing_metadata_tracked_as_failed(mock_session, mock_pdf, mock_fetch):
    rec = MagicMock()
    rec.record_id = 'rec-1'
    rec.record.doi = None
    rec.published_record = {}  # no metadata_records, no metadata
    mock_session.return_value = _mock_session([rec])
    mock_pdf.return_value = BytesIO(b'%PDF-fake')

    result = bundle_catalog_records(['rec-1'], USER_DATA)

    assert result.failed_count == 1
    assert result.failed[0]['reason'] == 'metadata_missing'
    os.unlink(result.zip_path)


@patch('odp.lib.record_dataset_bundler.fetch_external_file', return_value=(None, None))
@patch('odp.lib.record_dataset_bundler.generate_pdf')
@patch('odp.lib.record_dataset_bundler.Session')
def test_bundle_multiple_records(mock_session, mock_pdf, mock_fetch):
    records = [
        _mock_catalog_record(f'rec-{i}', f'10.1234/rec{i}', SAMPLE_METADATA)
        for i in range(3)
    ]
    mock_session.return_value = _mock_session(records)
    mock_pdf.return_value = BytesIO(b'%PDF-fake')

    result = bundle_catalog_records([f'rec-{i}' for i in range(3)], USER_DATA)

    assert result.record_count == 3
    assert result.failed_count == 0
    os.unlink(result.zip_path)
