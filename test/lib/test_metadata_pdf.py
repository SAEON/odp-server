"""
Comprehensive test suite for PDF generation and metadata adaptation modules.

Tests cover:
- PDF generation with various metadata inputs
- DataCite4 adapter with valid, minimal, and invalid data
- ISO19115 adapter with valid, minimal, and invalid data
- Schema auto-detection
- Fallback behavior
- Edge cases and error handling
- Special characters and Unicode handling
- Geographic and temporal extent handling

Coverage target: >90%
"""

import pytest
from io import BytesIO

from odp.lib.pdf_generator import (
    RecordMetadata,
    PersonInfo,
    GeographicExtent,
    TemporalExtent,
    License,
    generate_pdf,
)
from odp.lib.metadata_adapters import (
    DataCiteAdapter,
    ISO19115Adapter,
    adapt_metadata,
)
# Test Fixtures - Common test data

@pytest.fixture
def complete_datacite_metadata():
    """Complete DataCite4 metadata with all optional fields."""
    return {
        "titles": [{"title": "Marine Dataset: Ocean Temperature"}],
        "doi": "10.15493/marine",
        "publisher": "SAEON",
        "publicationYear": 2024,
        "creators": [
            {
                "name": "Dr. John Smith",
                "affiliation": [{"affiliation": "University of Cape Town, email: john@uct.ac.za"}],
                "nameIdentifiers": [{"nameIdentifierScheme": "ORCID", "nameIdentifier": "0000-0001-2345-6789"}],
            }
        ],
        "descriptions": [
            {
                "description": "This dataset contains ocean temperature measurements from 2020-2023.",
                "descriptionType": "Abstract",
            }
        ],
        "keywords": ["ocean", "temperature", "climate"],
        "geoLocations": [
            {
                "geoLocationBox": {
                    "northBoundLatitude": -20.0,
                    "southBoundLatitude": -35.0,
                    "eastBoundLongitude": 35.0,
                    "westBoundLongitude": 10.0,
                }
            }
        ],
        "rightsList": [
            {
                "rights": "CC BY 4.0",
                "rightsURI": "https://creativecommons.org/licenses/by/4.0/",
            }
        ],
        "contributors": [
            {
                "name": "Jane Doe",
                "contributorType": "ContactPerson",
                "affiliation": [{"affiliation": "SAEON, email: jane@saeon.ac.za"}],
            }
        ],
    }


@pytest.fixture
def minimal_datacite_metadata():
    """Minimal DataCite4 metadata with only required fields."""
    return {
        "titles": [{"title": "Minimal Dataset"}],
        "doi": "10.15493/minimal",
        "publisher": "Publisher",
        "publicationYear": 2024,
        "creators": [{"name": "Creator Name"}],
    }


@pytest.fixture
def complete_iso19115_metadata():
    """Complete ISO19115 metadata with all optional fields."""
    return {
        "title": "Marine Data: ISO Format",
        "fileIdentifier": "urn:uuid:12345678-1234-5678-1234-567812345678",
        "abstract": "A comprehensive marine dataset in ISO19115 format.",
        "responsibleParties": [
            {
                "role": "originator",
                "individualName": "Dr. Alice Johnson",
                "organizationName": "Marine Institute",
                "contactInfo": "email: alice@marine.org",
            },
            {
                "role": "publisher",
                "organizationName": "SAEON",
            },
            {
                "role": "pointOfContact",
                "individualName": "Bob Contact",
                "organizationName": "SAEON Support",
                "contactInfo": "email: bob@saeon.ac.za",
            },
        ],
        "keywords": ["marine", "biodiversity", "conservation"],
        "constraints": [
            {
                "rights": "All rights reserved",
                "rightsURI": "https://example.com/rights",
            }
        ],
        "extent": {
            "geographicElements": [
                {
                    "boundingBox": {
                        "northBoundLatitude": -22.0,
                        "southBoundLatitude": -34.0,
                        "eastBoundLongitude": 33.0,
                        "westBoundLongitude": 12.0,
                    }
                }
            ]
        },
    }


@pytest.fixture
def minimal_iso19115_metadata():
    """Minimal ISO19115 metadata with only required fields."""
    return {
        "title": "Minimal ISO Record",
        "fileIdentifier": "urn:uuid:87654321-4321-8765-4321-876543218765",
        "responsibleParties": [],
    }


@pytest.fixture
def record_metadata():
    """Complete RecordMetadata object for PDF testing."""
    return RecordMetadata(
        title="Test Dataset",
        doi="10.15493/test",
        publisher="Test Publisher",
        publication_year="2024",
        abstract="This is a test abstract with some details.",
        keywords=["test", "data", "example"],
        creator=PersonInfo(
            name="Test Author",
            affiliation="Test University",
            email="test@example.com",
            orcid="0000-0000-0000-0001",
        ),
        contact=PersonInfo(
            name="Support Person",
            affiliation="Support Org",
            email="support@example.com",
        ),
        license=License(text="CC BY 4.0", uri="https://creativecommons.org/licenses/by/4.0/"),
        geography=GeographicExtent(north=-20.0, south=-35.0, east=35.0, west=10.0),
        temporal=TemporalExtent(start_date="2020-01-01", end_date="2023-12-31"),
    )


# Tests for PDF Generation

class TestPDFGeneration:
    """Test suite for PDF generation functionality."""

    def test_generate_pdf_with_complete_metadata(self, record_metadata):
        """Test PDF generation with complete metadata."""
        pdf_buffer = generate_pdf(record_metadata)

        assert isinstance(pdf_buffer, BytesIO)
        assert pdf_buffer.tell() == 0  # Position at start

        pdf_content = pdf_buffer.getvalue()
        assert len(pdf_content) > 0
        assert pdf_content.startswith(b"%PDF")  # Valid PDF header

    def test_generate_pdf_with_minimal_metadata(self):
        """Test PDF generation with minimal metadata."""
        minimal = RecordMetadata(
            title="Minimal",
            doi="10.15493/min",
            publisher="Pub",
            publication_year="2024",
        )
        pdf_buffer = generate_pdf(minimal)

        assert isinstance(pdf_buffer, BytesIO)
        pdf_content = pdf_buffer.getvalue()
        assert pdf_content.startswith(b"%PDF")

    def test_generate_pdf_with_empty_keywords(self):
        """Test PDF generation when keywords are empty."""
        metadata = RecordMetadata(
            title="No Keywords",
            doi="10.15493/nk",
            publisher="Pub",
            publication_year="2024",
            keywords=[],
        )
        pdf_buffer = generate_pdf(metadata)

        pdf_content = pdf_buffer.getvalue()
        assert pdf_content.startswith(b"%PDF")

    def test_generate_pdf_with_special_characters(self):
        """Test PDF generation with special characters and Unicode."""
        metadata = RecordMetadata(
            title="Dataset with Spëcíål Çhäractérs",
            doi="10.15493/special",
            publisher="Éditions Françaises",
            publication_year="2024",
            abstract="Ábstract with åccénts and émojis 🌊",
            keywords=["café", "naïve", "résumé"],
        )
        pdf_buffer = generate_pdf(metadata)

        pdf_content = pdf_buffer.getvalue()
        assert pdf_content.startswith(b"%PDF")

    def test_generate_pdf_with_long_text(self):
        """Test PDF generation with very long text fields."""
        long_title = "A" * 500
        long_abstract = "B" * 2000

        metadata = RecordMetadata(
            title=long_title,
            doi="10.15493/long",
            publisher="Publisher",
            publication_year="2024",
            abstract=long_abstract,
        )
        pdf_buffer = generate_pdf(metadata)

        pdf_content = pdf_buffer.getvalue()
        assert pdf_content.startswith(b"%PDF")

    def test_generate_pdf_with_null_geographic_extent(self):
        """Test PDF generation with null geographic extent."""
        metadata = RecordMetadata(
            title="No Geography",
            doi="10.15493/nogeo",
            publisher="Pub",
            publication_year="2024",
            geography=None,
        )
        pdf_buffer = generate_pdf(metadata)

        pdf_content = pdf_buffer.getvalue()
        assert pdf_content.startswith(b"%PDF")

    def test_generate_pdf_with_zero_coordinates(self):
        """Test PDF generation with zero geographic coordinates."""
        metadata = RecordMetadata(
            title="Zero Coords",
            doi="10.15493/zero",
            publisher="Pub",
            publication_year="2024",
            geography=GeographicExtent(north=0.0, south=0.0, east=0.0, west=0.0),
        )
        pdf_buffer = generate_pdf(metadata)

        pdf_content = pdf_buffer.getvalue()
        assert pdf_content.startswith(b"%PDF")

    def test_generate_pdf_returns_readable_buffer(self, record_metadata):
        """Test that returned PDF buffer is readable."""
        pdf_buffer = generate_pdf(record_metadata)

        # Verify buffer is seekable and readable
        assert pdf_buffer.seekable()
        pdf_content = pdf_buffer.read()
        assert len(pdf_content) > 0

        # Verify we can seek back
        pdf_buffer.seek(0)
        assert pdf_buffer.tell() == 0

    def test_generate_pdf_error_handling(self):
        """Test PDF generation with invalid metadata type."""
        with pytest.raises((ValueError, AttributeError)):
            generate_pdf(None)


# Tests for DataCite Adapter

class TestDataCiteAdapter:
    """Test suite for DataCite4 metadata adapter."""

    def test_datacite_can_handle_valid_metadata(self, complete_datacite_metadata):
        """Test adapter detection of valid DataCite metadata."""
        adapter = DataCiteAdapter()
        assert adapter.can_handle(complete_datacite_metadata) is True

    def test_datacite_cannot_handle_missing_doi(self, complete_datacite_metadata):
        """Test adapter rejects metadata without DOI."""
        metadata = complete_datacite_metadata.copy()
        del metadata["doi"]

        adapter = DataCiteAdapter()
        assert adapter.can_handle(metadata) is False

    def test_datacite_cannot_handle_missing_titles(self, complete_datacite_metadata):
        """Test adapter rejects metadata without titles."""
        metadata = complete_datacite_metadata.copy()
        del metadata["titles"]

        adapter = DataCiteAdapter()
        assert adapter.can_handle(metadata) is False

    def test_datacite_cannot_handle_missing_creators(self, complete_datacite_metadata):
        """Test adapter rejects metadata without creators."""
        metadata = complete_datacite_metadata.copy()
        del metadata["creators"]

        adapter = DataCiteAdapter()
        assert adapter.can_handle(metadata) is False

    def test_datacite_adapt_complete_metadata(self, complete_datacite_metadata):
        """Test adaptation of complete DataCite metadata."""
        adapter = DataCiteAdapter()
        result = adapter.adapt(complete_datacite_metadata)

        assert isinstance(result, RecordMetadata)
        assert result.title == "Marine Dataset: Ocean Temperature"
        assert result.doi == "10.15493/marine"
        assert result.publisher == "SAEON"
        assert result.publication_year == "2024"
        assert result.abstract == "This dataset contains ocean temperature measurements from 2020-2023."
        assert result.keywords == ["ocean", "temperature", "climate"]
        assert result.creator.name == "Dr. John Smith"
        assert "john@uct.ac.za" in result.creator.email
        assert result.creator.orcid == "0000-0001-2345-6789"
        assert result.contact.name == "Jane Doe"
        assert "jane@saeon.ac.za" in result.contact.email
        assert result.license.text == "CC BY 4.0"
        assert result.geography.north == -20.0

    def test_datacite_adapt_minimal_metadata(self, minimal_datacite_metadata):
        """Test adaptation of minimal DataCite metadata."""
        adapter = DataCiteAdapter()
        result = adapter.adapt(minimal_datacite_metadata)

        assert result.title == "Minimal Dataset"
        assert result.doi == "10.15493/minimal"
        assert result.creator.name == "Creator Name"
        assert result.creator.affiliation == "N/A"
        assert result.abstract == "N/A"

    def test_datacite_missing_title_in_titles_array(self):
        """Test adapter with missing title in titles array."""
        metadata = {
            "titles": [{}],
            "doi": "10.15493/test",
            "publisher": "Pub",
            "publicationYear": 2024,
            "creators": [{"name": "Author"}],
        }
        adapter = DataCiteAdapter()
        result = adapter.adapt(metadata)
        assert result.title == "N/A"

    def test_datacite_empty_titles_array(self):
        """Test adapter with empty titles array."""
        metadata = {
            "titles": [],
            "doi": "10.15493/test",
            "publisher": "Pub",
            "publicationYear": 2024,
            "creators": [{"name": "Author"}],
        }
        adapter = DataCiteAdapter()
        result = adapter.adapt(metadata)
        assert result.title == "N/A"

    def test_datacite_keywords_as_string(self):
        """Test adapter with keywords as string instead of list."""
        metadata = {
            "titles": [{"title": "Test"}],
            "doi": "10.15493/test",
            "publisher": "Pub",
            "publicationYear": 2024,
            "creators": [{"name": "Author"}],
            "keywords": "keyword",
        }
        adapter = DataCiteAdapter()
        result = adapter.adapt(metadata)
        assert result.keywords == ["keyword"]

    def test_datacite_filter_empty_keywords(self):
        """Test adapter filters empty keyword strings."""
        metadata = {
            "titles": [{"title": "Test"}],
            "doi": "10.15493/test",
            "publisher": "Pub",
            "publicationYear": 2024,
            "creators": [{"name": "Author"}],
            "keywords": ["ocean", "", "data", ""],
        }
        adapter = DataCiteAdapter()
        result = adapter.adapt(metadata)
        assert result.keywords == ["ocean", "data"]

    def test_datacite_extract_email_from_affiliation(self):
        """Test email extraction from affiliation string."""
        metadata = {
            "titles": [{"title": "Test"}],
            "doi": "10.15493/test",
            "publisher": "Pub",
            "publicationYear": 2024,
            "creators": [
                {
                    "name": "Author",
                    "affiliation": [{"affiliation": "University, email: author@uni.edu"}],
                }
            ],
        }
        adapter = DataCiteAdapter()
        result = adapter.adapt(metadata)
        assert result.creator.email == "author@uni.edu"

    def test_datacite_geographic_extent_extraction(self):
        """Test geographic extent extraction."""
        metadata = {
            "titles": [{"title": "Test"}],
            "doi": "10.15493/test",
            "publisher": "Pub",
            "publicationYear": 2024,
            "creators": [{"name": "Author"}],
            "geoLocations": [
                {
                    "geoLocationBox": {
                        "northBoundLatitude": 10.5,
                        "southBoundLatitude": -5.5,
                        "eastBoundLongitude": 40.0,
                        "westBoundLongitude": 20.0,
                    }
                }
            ],
        }
        adapter = DataCiteAdapter()
        result = adapter.adapt(metadata)
        assert result.geography is not None
        assert result.geography.north == 10.5
        assert result.geography.south == -5.5

    def test_datacite_with_invalid_publication_year(self):
        """Test adapter with non-numeric publication year."""
        metadata = {
            "titles": [{"title": "Test"}],
            "doi": "10.15493/test",
            "publisher": "Pub",
            "publicationYear": "not-a-year",
            "creators": [{"name": "Author"}],
        }
        adapter = DataCiteAdapter()
        result = adapter.adapt(metadata)
        assert result.publication_year == "not-a-year"


# Tests for ISO19115 Adapter

class TestISO19115Adapter:
    """Test suite for ISO19115 metadata adapter."""

    def test_iso_can_handle_valid_metadata(self, complete_iso19115_metadata):
        """Test adapter detection of valid ISO19115 metadata."""
        adapter = ISO19115Adapter()
        assert adapter.can_handle(complete_iso19115_metadata) is True

    def test_iso_cannot_handle_missing_title(self, complete_iso19115_metadata):
        """Test adapter rejects metadata without title."""
        metadata = complete_iso19115_metadata.copy()
        del metadata["title"]

        adapter = ISO19115Adapter()
        assert adapter.can_handle(metadata) is False

    def test_iso_cannot_handle_missing_file_identifier(self, complete_iso19115_metadata):
        """Test adapter rejects metadata without fileIdentifier."""
        metadata = complete_iso19115_metadata.copy()
        del metadata["fileIdentifier"]

        adapter = ISO19115Adapter()
        assert adapter.can_handle(metadata) is False

    def test_iso_cannot_handle_missing_responsible_parties(self, complete_iso19115_metadata):
        """Test adapter rejects metadata without responsibleParties."""
        metadata = complete_iso19115_metadata.copy()
        del metadata["responsibleParties"]

        adapter = ISO19115Adapter()
        assert adapter.can_handle(metadata) is False

    def test_iso_adapt_complete_metadata(self, complete_iso19115_metadata):
        """Test adaptation of complete ISO19115 metadata."""
        adapter = ISO19115Adapter()
        result = adapter.adapt(complete_iso19115_metadata)

        assert isinstance(result, RecordMetadata)
        assert result.title == "Marine Data: ISO Format"
        assert result.doi == "urn:uuid:12345678-1234-5678-1234-567812345678"
        assert result.publisher == "SAEON"
        assert result.abstract == "A comprehensive marine dataset in ISO19115 format."
        assert result.keywords == ["marine", "biodiversity", "conservation"]
        assert result.creator.name == "Dr. Alice Johnson"
        assert result.creator.affiliation == "Marine Institute"
        assert "alice@marine.org" in result.creator.email
        assert result.contact.name == "Bob Contact"
        assert result.license.text == "All rights reserved"
        assert result.geography.north == -22.0

    def test_iso_adapt_minimal_metadata(self, minimal_iso19115_metadata):
        """Test adaptation of minimal ISO19115 metadata."""
        adapter = ISO19115Adapter()
        result = adapter.adapt(minimal_iso19115_metadata)

        assert result.title == "Minimal ISO Record"
        assert result.doi == "urn:uuid:87654321-4321-8765-4321-876543218765"
        assert result.publisher == "N/A"
        assert result.publication_year == "N/A"

    def test_iso_missing_abstract(self, minimal_iso19115_metadata):
        """Test adapter with missing abstract."""
        adapter = ISO19115Adapter()
        result = adapter.adapt(minimal_iso19115_metadata)
        assert result.abstract == "N/A"

    def test_iso_keywords_as_string(self):
        """Test adapter with keywords as string."""
        metadata = {
            "title": "Test",
            "fileIdentifier": "id123",
            "responsibleParties": [],
            "keywords": "keyword",
        }
        adapter = ISO19115Adapter()
        result = adapter.adapt(metadata)
        assert result.keywords == ["keyword"]

    def test_iso_empty_responsible_parties(self):
        """Test adapter with empty responsibleParties."""
        metadata = {
            "title": "Test",
            "fileIdentifier": "id123",
            "responsibleParties": [],
        }
        adapter = ISO19115Adapter()
        result = adapter.adapt(metadata)
        assert result.creator.name == "N/A"
        assert result.publisher == "N/A"
        assert result.contact.name == "N/A"

    def test_iso_extract_publisher_from_parties(self):
        """Test publisher extraction from responsible parties."""
        metadata = {
            "title": "Test",
            "fileIdentifier": "id123",
            "responsibleParties": [
                {"role": "publisher", "organizationName": "Test Publisher"}
            ],
        }
        adapter = ISO19115Adapter()
        result = adapter.adapt(metadata)
        assert result.publisher == "Test Publisher"

    def test_iso_extract_originator_and_contact(self):
        """Test extraction of originator and point of contact."""
        metadata = {
            "title": "Test",
            "fileIdentifier": "id123",
            "responsibleParties": [
                {
                    "role": "originator",
                    "individualName": "Originator",
                    "organizationName": "Orig Org",
                },
                {
                    "role": "pointOfContact",
                    "individualName": "Contact",
                    "organizationName": "Contact Org",
                },
            ],
        }
        adapter = ISO19115Adapter()
        result = adapter.adapt(metadata)
        assert result.creator.name == "Originator"
        assert result.contact.name == "Contact"

    def test_iso_geographic_extent_extraction(self):
        """Test geographic extent extraction."""
        metadata = {
            "title": "Test",
            "fileIdentifier": "id123",
            "responsibleParties": [],
            "extent": {
                "geographicElements": [
                    {
                        "boundingBox": {
                            "northBoundLatitude": 5.0,
                            "southBoundLatitude": -15.0,
                            "eastBoundLongitude": 30.0,
                            "westBoundLongitude": 15.0,
                        }
                    }
                ]
            },
        }
        adapter = ISO19115Adapter()
        result = adapter.adapt(metadata)
        assert result.geography is not None
        assert result.geography.north == 5.0
        assert result.geography.south == -15.0


# Tests for Auto-Detection Adapter

class TestAutoDetectAdapter:
    """Test suite for adapt_metadata auto-detection."""

    def test_auto_detect_datacite(self, complete_datacite_metadata):
        result = adapt_metadata(complete_datacite_metadata)
        assert result.title == "Marine Dataset: Ocean Temperature"
        assert result.doi == "10.15493/marine"

    def test_auto_detect_iso19115(self, complete_iso19115_metadata):
        result = adapt_metadata(complete_iso19115_metadata)
        assert result.title == "Marine Data: ISO Format"

    def test_auto_detect_invalid_format(self):
        with pytest.raises(ValueError, match="Could not detect or adapt metadata schema"):
            adapt_metadata({"unknown": "format", "fields": "here"})

    def test_auto_detect_empty_metadata(self):
        with pytest.raises(ValueError):
            adapt_metadata({})


# Tests for Factory Function

class TestAdaptMetadataFactory:
    """Test suite for adapt_metadata factory function."""

    def test_factory_auto_detect_datacite(self, complete_datacite_metadata):
        """Test factory with auto-detection on DataCite."""
        result = adapt_metadata(complete_datacite_metadata)
        assert result.title == "Marine Dataset: Ocean Temperature"

    def test_factory_auto_detect_iso19115(self, complete_iso19115_metadata):
        """Test factory with auto-detection on ISO19115."""
        result = adapt_metadata(complete_iso19115_metadata)
        assert result.title == "Marine Data: ISO Format"

    def test_factory_explicit_datacite(self, complete_datacite_metadata):
        """Test factory with explicit DataCite schema."""
        result = adapt_metadata(complete_datacite_metadata, schema_id="SAEON.DataCite4")
        assert result.doi == "10.15493/marine"

    def test_factory_explicit_datacite_shorthand(self, complete_datacite_metadata):
        """Test factory with short DataCite schema ID."""
        result = adapt_metadata(complete_datacite_metadata, schema_id="datacite4")
        assert result.doi == "10.15493/marine"

    def test_factory_explicit_iso19115(self, complete_iso19115_metadata):
        """Test factory with explicit ISO19115 schema."""
        result = adapt_metadata(complete_iso19115_metadata, schema_id="SAEON.ISO19115")
        assert result.title == "Marine Data: ISO Format"

    def test_factory_explicit_iso19115_shorthand(self, complete_iso19115_metadata):
        """Test factory with short ISO19115 schema ID."""
        result = adapt_metadata(complete_iso19115_metadata, schema_id="iso19115")
        assert result.title == "Marine Data: ISO Format"

    def test_factory_unknown_schema_with_fallback(self, complete_datacite_metadata):
        """Test factory with unknown schema but fallback enabled."""
        result = adapt_metadata(complete_datacite_metadata, schema_id="unknown_schema", fallback=True)
        # Should fallback to auto-detection and succeed
        assert result.doi == "10.15493/marine"

    def test_factory_unknown_schema_without_fallback(self, complete_datacite_metadata):
        """Test factory with unknown schema and fallback disabled."""
        with pytest.raises(ValueError, match="Unknown schema_id"):
            adapt_metadata(complete_datacite_metadata, schema_id="unknown_schema", fallback=False)

    def test_factory_none_schema_id(self, complete_datacite_metadata):
        """Test factory with None schema_id uses auto-detection."""
        result = adapt_metadata(complete_datacite_metadata, schema_id=None)
        assert result.doi == "10.15493/marine"

    def test_factory_explicit_schema_with_fallback(self, complete_iso19115_metadata):
        """Test factory that tries explicit schema then falls back."""
        # Provide DataCite schema ID but ISO19115 data - should fallback
        result = adapt_metadata(
            complete_iso19115_metadata,
            schema_id="SAEON.DataCite4",
            fallback=True
        )
        # Should succeed via fallback to auto-detection
        assert result.title == "Marine Data: ISO Format"

    def test_factory_explicit_schema_without_fallback_fails(self):
        """Test factory fails when explicit schema doesn't match and no fallback."""
        iso_data = {
            "title": "ISO",
            "fileIdentifier": "id",
            "responsibleParties": [],
        }

        # Try to adapt ISO as DataCite without fallback
        with pytest.raises(ValueError):
            adapt_metadata(iso_data, schema_id="SAEON.DataCite4", fallback=False)


# Tests for Data Classes

class TestDataClasses:
    """Test suite for data classes."""

    def test_person_info_defaults(self):
        """Test PersonInfo default values."""
        person = PersonInfo()
        assert person.name == "N/A"
        assert person.affiliation == "N/A"
        assert person.email == "N/A"
        assert person.orcid == "N/A"

    def test_person_info_initialization(self):
        """Test PersonInfo with custom values."""
        person = PersonInfo(name="John", affiliation="Org", email="john@org.com")
        assert person.name == "John"
        assert person.affiliation == "Org"
        assert person.email == "john@org.com"

    def test_geographic_extent_string_representation(self):
        """Test GeographicExtent string formatting."""
        geo = GeographicExtent(north=10, south=-10, east=40, west=20)
        geo_str = str(geo)

        assert "North: 10" in geo_str
        assert "South: -10" in geo_str
        assert "East: 40" in geo_str
        assert "West: 20" in geo_str

    def test_temporal_extent_string_representation(self):
        """Test TemporalExtent string formatting."""
        temp = TemporalExtent(start_date="2020-01-01", end_date="2023-12-31")
        temp_str = str(temp)

        assert "2020-01-01" in temp_str
        assert "2023-12-31" in temp_str
        assert "–" in temp_str

    def test_license_without_uri(self):
        """Test License without URI returns text only."""
        license_info = License(text="CC BY 4.0")
        html = license_info.to_html()
        assert html == "CC BY 4.0"

    def test_license_with_uri(self):
        """Test License with URI returns HTML link."""
        license_info = License(
            text="CC BY 4.0",
            uri="https://creativecommons.org/licenses/by/4.0/"
        )
        html = license_info.to_html()
        assert "link href=" in html
        assert "CC BY 4.0" in html

    def test_record_metadata_post_init_defaults(self):
        """Test RecordMetadata __post_init__ sets defaults."""
        metadata = RecordMetadata(
            title="Test",
            doi="10.15493/test",
            publisher="Pub",
            publication_year="2024",
        )

        assert metadata.keywords == []
        assert isinstance(metadata.creator, PersonInfo)
        assert isinstance(metadata.contact, PersonInfo)
        assert isinstance(metadata.license, License)
        assert metadata.geography is None
        assert isinstance(metadata.temporal, TemporalExtent)

    def test_record_metadata_explicit_values(self):
        """Test RecordMetadata preserves explicit values."""
        keywords = ["test", "data"]
        creator = PersonInfo(name="Author")

        metadata = RecordMetadata(
            title="Test",
            doi="10.15493/test",
            publisher="Pub",
            publication_year="2024",
            keywords=keywords,
            creator=creator,
        )

        assert metadata.keywords is keywords
        assert metadata.creator is creator


# Integration Tests

class TestIntegration:
    """Integration tests for the full PDF generation pipeline."""

    def test_full_pipeline_datacite(self, complete_datacite_metadata):
        """Test complete pipeline: DataCite -> adapt -> PDF."""
        normalized = adapt_metadata(complete_datacite_metadata)
        pdf_buffer = generate_pdf(normalized)

        pdf_content = pdf_buffer.getvalue()
        assert pdf_content.startswith(b"%PDF")
        assert len(pdf_content) > 100

    def test_full_pipeline_iso19115(self, complete_iso19115_metadata):
        """Test complete pipeline: ISO19115 -> adapt -> PDF."""
        normalized = adapt_metadata(complete_iso19115_metadata)
        pdf_buffer = generate_pdf(normalized)

        pdf_content = pdf_buffer.getvalue()
        assert pdf_content.startswith(b"%PDF")

    def test_full_pipeline_with_explicit_schema(self, complete_datacite_metadata):
        """Test complete pipeline with explicit schema specification."""
        normalized = adapt_metadata(
            complete_datacite_metadata,
            schema_id="SAEON.DataCite4"
        )
        pdf_buffer = generate_pdf(normalized)

        assert pdf_buffer.getvalue().startswith(b"%PDF")

    def test_multiple_pdfs_from_same_adapter(self, complete_datacite_metadata, minimal_datacite_metadata):
        """Test generating multiple PDFs with same adapter."""
        pdf1 = generate_pdf(adapt_metadata(complete_datacite_metadata))
        pdf2 = generate_pdf(adapt_metadata(minimal_datacite_metadata))

        assert pdf1.getvalue() != pdf2.getvalue()
        assert len(pdf1.getvalue()) > len(pdf2.getvalue())

    def test_concurrent_adaptations_different_schemas(
        self,
        complete_datacite_metadata,
        complete_iso19115_metadata
    ):
        """Test adapting different schemas produces consistent results."""
        dc_result = adapt_metadata(complete_datacite_metadata)
        iso_result = adapt_metadata(complete_iso19115_metadata)

        # Both should be valid RecordMetadata
        assert isinstance(dc_result, RecordMetadata)
        assert isinstance(iso_result, RecordMetadata)

        # Both should have required fields
        assert dc_result.title and dc_result.doi
        assert iso_result.title and iso_result.doi


# Edge Cases and Error Scenarios

class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_extremely_large_coordinates(self):
        """Test geographic extent with extreme values."""
        metadata = RecordMetadata(
            title="Extreme",
            doi="10.15493/extreme",
            publisher="Pub",
            publication_year="2024",
            geography=GeographicExtent(
                north=90.0, south=-90.0,
                east=180.0, west=-180.0
            ),
        )
        pdf_buffer = generate_pdf(metadata)
        assert pdf_buffer.getvalue().startswith(b"%PDF")

    def test_metadata_with_null_bytes(self):
        """Test handling of null bytes in text fields."""
        metadata = RecordMetadata(
            title="Title\x00with\x00nulls",
            doi="10.15493/null",
            publisher="Pub",
            publication_year="2024",
        )
        # Should not raise
        pdf_buffer = generate_pdf(metadata)
        assert pdf_buffer.getvalue().startswith(b"%PDF")

    def test_many_keywords(self):
        """Test metadata with many keywords."""
        keywords = [f"keyword_{i}" for i in range(100)]
        metadata = RecordMetadata(
            title="Many",
            doi="10.15493/many",
            publisher="Pub",
            publication_year="2024",
            keywords=keywords,
        )
        pdf_buffer = generate_pdf(metadata)
        assert pdf_buffer.getvalue().startswith(b"%PDF")

    def test_adapter_with_malformed_geographic_extent(self):
        """Test adapter handles malformed geographic extent gracefully."""
        metadata = {
            "titles": [{"title": "Test"}],
            "doi": "10.15493/test",
            "publisher": "Pub",
            "publicationYear": 2024,
            "creators": [{"name": "Author"}],
            "geoLocations": [{"geoLocationBox": {"invalid": "data"}}],
        }
        adapter = DataCiteAdapter()
        # Should not raise, just skip geography
        result = adapter.adapt(metadata)
        assert result.geography is None

    def test_very_long_doi(self):
        """Test handling of very long DOI."""
        long_doi = "10." + "1" * 500
        metadata = RecordMetadata(
            title="Long DOI",
            doi=long_doi,
            publisher="Pub",
            publication_year="2024",
        )
        pdf_buffer = generate_pdf(metadata)
        assert pdf_buffer.getvalue().startswith(b"%PDF")
