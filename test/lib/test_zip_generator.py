"""
Unit tests for ZIP bundle generation helper module.

Tests cover:
- Folder name sanitization (invalid chars, spaces, length)
- External file downloading (success, timeouts, errors)
- Record fetching by DOI
- Error handling and logging

Coverage target: >95%
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from odp.lib.zip_generator import (
    create_safe_folder_name,
    fetch_external_file,
    fetch_record_by_doi,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_requests_get():
    """Mock requests.get for external file downloads."""
    with patch('odp.lib.zip_generator.requests.get') as mock:
        yield mock


# ============================================================================
# Tests: create_safe_folder_name
# ============================================================================

class TestCreateSafeFolderName:
    """Tests for folder name sanitization."""

    def test_valid_simple_name(self):
        """Valid names should pass through unchanged."""
        assert create_safe_folder_name("Marine Dataset") == "Marine_Dataset"

    def test_removes_invalid_characters(self):
        """Should remove all invalid filesystem characters."""
        result = create_safe_folder_name("Test<>:File/Name\\Data|*?Data")
        assert '<' not in result
        assert '>' not in result
        assert ':' not in result
        assert '/' not in result
        assert '\\' not in result
        assert '|' not in result
        assert '*' not in result
        assert '?' not in result

    def test_replaces_spaces_with_underscores(self):
        """Spaces should be replaced with underscores."""
        assert create_safe_folder_name("Ocean Temperature Data") == "Ocean_Temperature_Data"

    def test_removes_consecutive_underscores(self):
        """Multiple consecutive underscores should be collapsed."""
        assert create_safe_folder_name("Test  Multiple   Spaces") == "Test_Multiple_Spaces"

    def test_strips_leading_trailing_underscores(self):
        """Leading/trailing underscores should be stripped."""
        assert create_safe_folder_name("  Spaced  ") == "Spaced"

    def test_truncates_long_names(self):
        """Names longer than max_length should be truncated."""
        long_name = "A" * 250
        result = create_safe_folder_name(long_name, max_length=200)
        assert len(result) <= 200
        assert not result.endswith('_')

    def test_handles_empty_string(self):
        """Empty strings should return 'Untitled'."""
        assert create_safe_folder_name("") == "Untitled"
        assert create_safe_folder_name("   ") == "Untitled"

    def test_handles_none_value(self):
        """None input should return 'Untitled'."""
        assert create_safe_folder_name(None) == "Untitled"

    def test_handles_special_characters_only(self):
        """Strings with only invalid chars should return 'Untitled'."""
        assert create_safe_folder_name("<>:/\\|?*") == "Untitled"

    def test_unicode_characters_preserved(self):
        """Unicode characters should be preserved."""
        result = create_safe_folder_name("Dataset_2024_αβγ")
        assert "αβγ" in result

    def test_complex_real_world_example(self):
        """Complex real-world title should be sanitized correctly."""
        title = 'Climate Data: 2020-2023 "Global" Analysis (v1.0)'
        result = create_safe_folder_name(title)
        assert result  # Should not be empty
        assert ':' not in result
        assert '"' not in result
        assert '/' not in result


# ============================================================================
# Tests: fetch_external_file
# ============================================================================

class TestFetchExternalFile:
    """Tests for external file downloading."""

    def test_successful_download(self, mock_requests_get):
        """Successful HTTP 200 should return file bytes."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.headers = {'Content-Length': '100'}
        mock_response.iter_content = Mock(return_value=[b'test', b'data'])
        mock_requests_get.return_value = mock_response

        result = fetch_external_file('https://example.com/file.csv')

        assert result == b'testdata'
        mock_requests_get.assert_called_once_with(
            'https://example.com/file.csv',
            timeout=30,
            stream=True
        )

    def test_empty_url(self, mock_requests_get):
        """Empty URL should return None."""
        result = fetch_external_file('')
        assert result is None
        mock_requests_get.assert_not_called()

    def test_none_url(self, mock_requests_get):
        """None URL should return None."""
        result = fetch_external_file(None)
        assert result is None
        mock_requests_get.assert_not_called()

    def test_http_404_error(self, mock_requests_get):
        """HTTP 404 should return None."""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 404
        mock_requests_get.return_value = mock_response

        result = fetch_external_file('https://example.com/notfound.csv')

        assert result is None

    def test_http_500_error(self, mock_requests_get):
        """HTTP 500 should return None."""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_requests_get.return_value = mock_response

        result = fetch_external_file('https://example.com/error.csv')

        assert result is None

    def test_timeout_error(self, mock_requests_get):
        """Timeout exception should return None."""
        mock_requests_get.side_effect = requests.Timeout()

        result = fetch_external_file('https://example.com/file.csv')

        assert result is None

    def test_connection_error(self, mock_requests_get):
        """Connection error should return None."""
        mock_requests_get.side_effect = requests.ConnectionError()

        result = fetch_external_file('https://example.com/file.csv')

        assert result is None

    def test_request_exception(self, mock_requests_get):
        """Generic request exception should return None."""
        mock_requests_get.side_effect = requests.RequestException()

        result = fetch_external_file('https://example.com/file.csv')

        assert result is None

    def test_custom_timeout(self, mock_requests_get):
        """Custom timeout should be passed to requests."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.headers = {}
        mock_response.iter_content = Mock(return_value=[b'data'])
        mock_requests_get.return_value = mock_response

        fetch_external_file('https://example.com/file.csv', timeout=60)

        mock_requests_get.assert_called_once()
        assert mock_requests_get.call_args.kwargs['timeout'] == 60

    def test_file_size_limit_on_header(self, mock_requests_get):
        """Content-Length exceeding max_size should return None."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.headers = {'Content-Length': '1000000'}  # 1MB
        mock_requests_get.return_value = mock_response

        result = fetch_external_file(
            'https://example.com/file.csv',
            max_size=500000  # 500KB limit
        )

        assert result is None

    def test_file_size_limit_during_download(self, mock_requests_get):
        """Exceeding max_size during download should return None."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.headers = {}
        # Simulate chunks that exceed limit
        mock_response.iter_content = Mock(
            return_value=[b'x' * 100000, b'y' * 100000, b'z' * 100000]
        )
        mock_requests_get.return_value = mock_response

        result = fetch_external_file(
            'https://example.com/file.csv',
            max_size=150000
        )

        assert result is None

    def test_empty_file(self, mock_requests_get):
        """Empty file response should return None."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.headers = {'Content-Length': '0'}
        mock_response.iter_content = Mock(return_value=[])
        mock_requests_get.return_value = mock_response

        result = fetch_external_file('https://example.com/empty.csv')

        assert result is None

    def test_chunked_large_file(self, mock_requests_get):
        """Large file with chunked download should work."""
        chunks = [b'chunk1', b'chunk2', b'chunk3']
        mock_response = Mock()
        mock_response.ok = True
        mock_response.headers = {}
        mock_response.iter_content = Mock(return_value=chunks)
        mock_requests_get.return_value = mock_response

        result = fetch_external_file('https://example.com/large.csv')

        assert result == b'chunk1chunk2chunk3'

    def test_invalid_content_length_header(self, mock_requests_get):
        """Invalid Content-Length header should not crash."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.headers = {'Content-Length': 'invalid'}
        mock_response.iter_content = Mock(return_value=[b'data'])
        mock_requests_get.return_value = mock_response

        result = fetch_external_file('https://example.com/file.csv', max_size=100)

        assert result == b'data'


# ============================================================================
# Tests: fetch_record_by_doi
# ============================================================================

class TestFetchRecordByDoi:
    """Tests for fetching records by DOI."""

    @patch('odp.lib.zip_generator.Session')
    @patch('odp.lib.zip_generator.select')
    def test_record_found(self, mock_select, mock_session_class):
        """Should return record when DOI matches."""
        # Create mock record
        mock_record_data = Mock()
        mock_record_data.data = {
            'metadata': {'doi': '10.15493/TEST', 'title': 'Test Dataset'}
        }

        mock_catalog_record = Mock()
        mock_catalog_record.record = mock_record_data
        mock_catalog_record.published = True

        # Setup session mock
        mock_session = MagicMock()
        mock_execute = Mock()
        mock_execute.scalars.return_value.all.return_value = [mock_catalog_record]
        mock_session.execute.return_value = mock_execute
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = fetch_record_by_doi('10.15493/TEST')

        assert result == mock_catalog_record

    @patch('odp.lib.zip_generator.Session')
    @patch('odp.lib.zip_generator.select')
    def test_record_not_found(self, mock_select, mock_session_class):
        """Should return None when DOI not found."""
        mock_session = MagicMock()
        mock_execute = Mock()
        mock_execute.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_execute
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = fetch_record_by_doi('10.15493/NONEXISTENT')

        assert result is None

    @patch('odp.lib.zip_generator.Session')
    @patch('odp.lib.zip_generator.select')
    def test_database_error(self, mock_select, mock_session_class):
        """Should return None on database error."""
        mock_session = MagicMock()
        mock_session.execute.side_effect = Exception('Database error')
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = fetch_record_by_doi('10.15493/TEST')

        assert result is None

    @patch('odp.lib.zip_generator.Session')
    @patch('odp.lib.zip_generator.select')
    def test_empty_doi_string(self, mock_select, mock_session_class):
        """Should handle empty DOI gracefully."""
        mock_session = MagicMock()
        mock_execute = Mock()
        mock_execute.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_execute
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = fetch_record_by_doi('')

        assert result is None

    @patch('odp.lib.zip_generator.Session')
    @patch('odp.lib.zip_generator.select')
    def test_multiple_records_returns_first_match(self, mock_select, mock_session_class):
        """Should return first record when multiple DOI matches exist."""
        mock_record1 = Mock()
        mock_record1.record.data = {'metadata': {'doi': '10.15493/TEST'}}

        mock_record2 = Mock()
        mock_record2.record.data = {'metadata': {'doi': '10.15493/TEST'}}

        mock_session = MagicMock()
        mock_execute = Mock()
        mock_execute.scalars.return_value.all.return_value = [mock_record1, mock_record2]
        mock_session.execute.return_value = mock_execute
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = fetch_record_by_doi('10.15493/TEST')

        assert result == mock_record1

    @patch('odp.lib.zip_generator.Session')
    @patch('odp.lib.zip_generator.select')
    def test_record_without_metadata(self, mock_select, mock_session_class):
        """Should handle records without metadata gracefully."""
        mock_record = Mock()
        mock_record.record.data = {}  # No 'metadata' key

        mock_session = MagicMock()
        mock_execute = Mock()
        mock_execute.scalars.return_value.all.return_value = [mock_record]
        mock_session.execute.return_value = mock_execute
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = fetch_record_by_doi('10.15493/TEST')

        assert result is None

    @patch('odp.lib.zip_generator.Session')
    @patch('odp.lib.zip_generator.select')
    def test_record_with_missing_doi_field(self, mock_select, mock_session_class):
        """Should skip records with no DOI in metadata."""
        mock_record = Mock()
        mock_record.record.data = {'metadata': {'title': 'Test'}}  # No DOI

        mock_session = MagicMock()
        mock_execute = Mock()
        mock_execute.scalars.return_value.all.return_value = [mock_record]
        mock_session.execute.return_value = mock_execute
        mock_session_class.return_value.__enter__.return_value = mock_session

        result = fetch_record_by_doi('10.15493/TEST')

        assert result is None


# ============================================================================
# Tests: create_zip_bundle (Main Function)
# ============================================================================

class TestCreateZipBundle:
    """Tests for complete ZIP bundle generation."""

    @patch('odp.lib.zip_generator.fetch_record_by_doi')
    @patch('odp.lib.zip_generator.adapt_metadata')
    @patch('odp.lib.zip_generator.generate_pdf')
    @patch('odp.lib.zip_generator.fetch_external_file')
    @patch('odp.lib.zip_generator.log_bundle_download_audit')
    def test_create_zip_bundle_success(
        self, mock_audit, mock_fetch_file, mock_gen_pdf,
        mock_adapt, mock_fetch_record
    ):
        """Should create ZIP with PDFs and files."""
        from io import BytesIO
        from zipfile import ZipFile

        # Setup mocks
        mock_record = Mock()
        mock_record.record.data = {
            'metadata': {
                'titles': [{'title': 'Test Dataset'}],
                'doi': '10.15493/TEST',
                'immutableResource': {
                    'resourceDownload': {
                        'downloadURL': 'https://example.com/data.csv',
                        'fileName': 'data.csv'
                    }
                }
            }
        }
        mock_fetch_record.return_value = mock_record
        mock_gen_pdf.return_value = BytesIO(b'PDF content')
        mock_fetch_file.return_value = b'CSV content'

        # Import and call function
        from odp.lib.zip_generator import create_zip_bundle

        zip_bytes, metadata = create_zip_bundle(
            record_ids=['10.15493/TEST'],
            user_data={
                'name': 'Test User',
                'email': 'test@example.com',
                'organisation': 'Test Org'
            },
            client_ip='192.168.1.1',
            user_agent='TestAgent'
        )

        # Assertions
        assert zip_bytes is not None
        assert len(zip_bytes) > 0
        assert metadata['record_count'] == 1
        assert metadata['failed_count'] == 0
        assert metadata['total_size'] > 0

        # Verify ZIP contents
        with ZipFile(BytesIO(zip_bytes)) as zf:
            files = zf.namelist()
            assert 'Test_Dataset/metadata.pdf' in files
            assert 'Test_Dataset/data.csv' in files

        # Verify audit was logged
        mock_audit.assert_called_once()

    def test_create_zip_bundle_missing_record_ids(self):
        """Should raise ValueError if record_ids empty."""
        from odp.lib.zip_generator import create_zip_bundle

        with pytest.raises(ValueError, match="record_ids cannot be empty"):
            create_zip_bundle(
                record_ids=[],
                user_data={
                    'name': 'Test',
                    'email': 'test@example.com',
                    'organisation': 'Test'
                }
            )

    def test_create_zip_bundle_missing_user_fields(self):
        """Should raise ValueError if user_data missing required fields."""
        from odp.lib.zip_generator import create_zip_bundle

        with pytest.raises(ValueError, match="user_data missing required fields"):
            create_zip_bundle(
                record_ids=['10.15493/TEST'],
                user_data={'name': 'Test'}  # Missing email and organisation
            )

    @patch('odp.lib.zip_generator.fetch_record_by_doi')
    @patch('odp.lib.zip_generator.log_bundle_download_audit')
    def test_create_zip_bundle_graceful_degradation(
        self, mock_audit, mock_fetch_record
    ):
        """Should handle partial failures gracefully."""
        # First record exists, second doesn't
        mock_record = Mock()
        mock_record.record.data = {
            'metadata': {
                'titles': [{'title': 'Found Dataset'}],
                'doi': '10.15493/FOUND',
                'immutableResource': {'resourceDownload': {}}
            }
        }
        mock_fetch_record.side_effect = [mock_record, None]

        from odp.lib.zip_generator import create_zip_bundle

        with patch('odp.lib.zip_generator.adapt_metadata') as mock_adapt:
            with patch('odp.lib.zip_generator.generate_pdf') as mock_gen_pdf:
                from io import BytesIO
                mock_gen_pdf.return_value = BytesIO(b'PDF')

                zip_bytes, metadata = create_zip_bundle(
                    record_ids=['10.15493/FOUND', '10.15493/MISSING'],
                    user_data={
                        'name': 'Test',
                        'email': 'test@example.com',
                        'organisation': 'Test'
                    }
                )

                # Should have 1 success and 1 failure
                assert metadata['record_count'] == 1
                assert metadata['failed_count'] == 1
                assert len(metadata['processed']) == 1
                assert len(metadata['failed']) == 1
