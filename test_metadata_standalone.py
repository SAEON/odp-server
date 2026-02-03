"""
Standalone test runner for PDF generation and metadata adaptation modules.

This file can be run without the full ODP environment setup.
Usage: python test_metadata_standalone.py
"""

import sys
from io import BytesIO

# Add the odp-server to path
sys.path.insert(0, '/home/n.bingani/Documents/Work/odp-build/odp-server')

from odp.lib.metadata_pdf import (
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
    AutoDetectAdapter,
    adapt_metadata,
)


def test_pdf_generation_basic():
    """Test basic PDF generation."""
    metadata = RecordMetadata(
        title="Test Dataset",
        doi="10.15493/test",
        publisher="Test Publisher",
        publication_year="2024",
    )
    pdf = generate_pdf(metadata)
    assert isinstance(pdf, BytesIO)
    assert pdf.getvalue().startswith(b"%PDF")
    print("✓ test_pdf_generation_basic")


def test_pdf_generation_complete():
    """Test PDF generation with complete metadata."""
    metadata = RecordMetadata(
        title="Marine Dataset",
        doi="10.15493/marine",
        publisher="SAEON",
        publication_year="2024",
        abstract="Ocean temperature data",
        keywords=["ocean", "temperature"],
        creator=PersonInfo(name="John Smith", email="john@example.com"),
        contact=PersonInfo(name="Jane Doe", email="jane@example.com"),
        license=License(text="CC BY 4.0", uri="https://creativecommons.org/licenses/by/4.0/"),
        geography=GeographicExtent(north=-20.0, south=-35.0, east=35.0, west=10.0),
        temporal=TemporalExtent(start_date="2020-01-01", end_date="2023-12-31"),
    )
    pdf = generate_pdf(metadata)
    assert isinstance(pdf, BytesIO)
    assert pdf.getvalue().startswith(b"%PDF")
    print("✓ test_pdf_generation_complete")


def test_datacite_adapter():
    """Test DataCite adapter."""
    metadata = {
        "titles": [{"title": "Marine Dataset"}],
        "doi": "10.15493/marine",
        "publisher": "SAEON",
        "publicationYear": 2024,
        "creators": [
            {
                "name": "Dr. John Smith",
                "affiliation": [{"affiliation": "University, email: john@example.com"}],
            }
        ],
        "descriptions": [
            {
                "description": "Ocean temperature data",
                "descriptionType": "Abstract",
            }
        ],
        "keywords": ["ocean", "data"],
    }

    adapter = DataCiteAdapter()
    assert adapter.can_handle(metadata)

    result = adapter.adapt(metadata)
    assert result.title == "Marine Dataset"
    assert result.doi == "10.15493/marine"
    assert result.publisher == "SAEON"
    assert result.creator.name == "Dr. John Smith"
    assert "john@example.com" in result.creator.email
    print("✓ test_datacite_adapter")


def test_iso19115_adapter():
    """Test ISO19115 adapter."""
    metadata = {
        "title": "Marine Data",
        "fileIdentifier": "urn:uuid:12345678",
        "abstract": "Ocean data",
        "responsibleParties": [
            {
                "role": "originator",
                "individualName": "Alice Johnson",
                "organizationName": "Marine Org",
            },
            {
                "role": "publisher",
                "organizationName": "SAEON",
            },
        ],
        "keywords": ["marine", "data"],
    }

    adapter = ISO19115Adapter()
    assert adapter.can_handle(metadata)

    result = adapter.adapt(metadata)
    assert result.title == "Marine Data"
    assert result.creator.name == "Alice Johnson"
    assert result.publisher == "SAEON"
    print("✓ test_iso19115_adapter")


def test_auto_detect_datacite():
    """Test auto-detection of DataCite."""
    metadata = {
        "titles": [{"title": "Dataset"}],
        "doi": "10.15493/test",
        "publisher": "Pub",
        "publicationYear": 2024,
        "creators": [{"name": "Author"}],
    }

    adapter = AutoDetectAdapter()
    result = adapter.adapt(metadata)
    assert result.title == "Dataset"
    print("✓ test_auto_detect_datacite")


def test_auto_detect_iso19115():
    """Test auto-detection of ISO19115."""
    metadata = {
        "title": "Dataset",
        "fileIdentifier": "id123",
        "responsibleParties": [],
    }

    adapter = AutoDetectAdapter()
    result = adapter.adapt(metadata)
    assert result.title == "Dataset"
    print("✓ test_auto_detect_iso19115")


def test_factory_function():
    """Test factory function."""
    metadata = {
        "titles": [{"title": "Dataset"}],
        "doi": "10.15493/test",
        "publisher": "Pub",
        "publicationYear": 2024,
        "creators": [{"name": "Author"}],
    }

    result = adapt_metadata(metadata)
    assert result.title == "Dataset"
    assert result.doi == "10.15493/test"
    print("✓ test_factory_function")


def test_factory_explicit_schema():
    """Test factory with explicit schema."""
    metadata = {
        "titles": [{"title": "Dataset"}],
        "doi": "10.15493/test",
        "publisher": "Pub",
        "publicationYear": 2024,
        "creators": [{"name": "Author"}],
    }

    result = adapt_metadata(metadata, schema_id="SAEON.DataCite4")
    assert result.title == "Dataset"
    print("✓ test_factory_explicit_schema")


def test_special_characters():
    """Test handling of special characters."""
    metadata = RecordMetadata(
        title="Dataset with Spëcíål Çhäractérs 🌊",
        doi="10.15493/special",
        publisher="Éditions Françaises",
        publication_year="2024",
    )
    pdf = generate_pdf(metadata)
    assert pdf.getvalue().startswith(b"%PDF")
    print("✓ test_special_characters")


def test_long_text():
    """Test handling of long text."""
    metadata = RecordMetadata(
        title="A" * 500,
        doi="10.15493/long",
        publisher="Pub",
        publication_year="2024",
        abstract="B" * 2000,
    )
    pdf = generate_pdf(metadata)
    assert pdf.getvalue().startswith(b"%PDF")
    print("✓ test_long_text")


def test_geographic_extent_formatting():
    """Test geographic extent string formatting."""
    geo = GeographicExtent(north=10.0, south=-10.0, east=40.0, west=20.0)
    geo_str = str(geo)
    assert "North: 10" in geo_str
    assert "South: -10" in geo_str
    print("✓ test_geographic_extent_formatting")


def test_temporal_extent_formatting():
    """Test temporal extent string formatting."""
    temp = TemporalExtent(start_date="2020-01-01", end_date="2023-12-31")
    temp_str = str(temp)
    assert "2020-01-01" in temp_str
    assert "2023-12-31" in temp_str
    print("✓ test_temporal_extent_formatting")


def test_license_without_uri():
    """Test license without URI."""
    license_info = License(text="CC BY 4.0")
    assert license_info.to_html() == "CC BY 4.0"
    print("✓ test_license_without_uri")


def test_license_with_uri():
    """Test license with URI."""
    license_info = License(
        text="CC BY 4.0",
        uri="https://creativecommons.org/licenses/by/4.0/"
    )
    html = license_info.to_html()
    assert "link href=" in html
    assert "CC BY 4.0" in html
    print("✓ test_license_with_uri")


def test_record_metadata_defaults():
    """Test RecordMetadata defaults."""
    metadata = RecordMetadata(
        title="Test",
        doi="10.15493/test",
        publisher="Pub",
        publication_year="2024",
    )
    assert metadata.keywords == []
    assert isinstance(metadata.creator, PersonInfo)
    assert isinstance(metadata.contact, PersonInfo)
    print("✓ test_record_metadata_defaults")


def test_datacite_missing_fields():
    """Test adapter with missing optional fields."""
    metadata = {
        "titles": [{"title": "Test"}],
        "doi": "10.15493/test",
        "publisher": "Pub",
        "publicationYear": 2024,
        "creators": [{"name": "Author"}],
    }
    adapter = DataCiteAdapter()
    result = adapter.adapt(metadata)
    assert result.abstract == "N/A"
    assert result.keywords == []
    print("✓ test_datacite_missing_fields")


def test_iso19115_missing_fields():
    """Test ISO adapter with missing fields."""
    metadata = {
        "title": "Test",
        "fileIdentifier": "id123",
        "responsibleParties": [],
    }
    adapter = ISO19115Adapter()
    result = adapter.adapt(metadata)
    assert result.publisher == "N/A"
    assert result.publication_year == "N/A"
    print("✓ test_iso19115_missing_fields")


def test_full_pipeline():
    """Test complete pipeline: adapt then generate PDF."""
    datacite = {
        "titles": [{"title": "Marine Dataset"}],
        "doi": "10.15493/marine",
        "publisher": "SAEON",
        "publicationYear": 2024,
        "creators": [{"name": "John Smith"}],
    }

    normalized = adapt_metadata(datacite)
    pdf = generate_pdf(normalized)
    assert pdf.getvalue().startswith(b"%PDF")
    print("✓ test_full_pipeline")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running standalone metadata PDF tests")
    print("="*60 + "\n")

    tests = [
        test_pdf_generation_basic,
        test_pdf_generation_complete,
        test_datacite_adapter,
        test_iso19115_adapter,
        test_auto_detect_datacite,
        test_auto_detect_iso19115,
        test_factory_function,
        test_factory_explicit_schema,
        test_special_characters,
        test_long_text,
        test_geographic_extent_formatting,
        test_temporal_extent_formatting,
        test_license_without_uri,
        test_license_with_uri,
        test_record_metadata_defaults,
        test_datacite_missing_fields,
        test_iso19115_missing_fields,
        test_full_pipeline,
    ]

    failed = []
    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {str(e)}")
            failed.append((test.__name__, e))

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {len(failed)} failed")
    print("="*60 + "\n")

    if failed:
        print("Failed tests:")
        for test_name, error in failed:
            print(f"  - {test_name}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
