"""
Metadata schema adapters for normalizing different metadata formats.

This module provides adapters to convert DataCite 4, ISO19115, and other
metadata schemas into the unified RecordMetadata format for PDF generation.

The adapter pattern allows:
- Support for multiple schemas without modifying PDF generation code
- Easy addition of new schemas
- Graceful fallback when schemas are partial or inconsistent

Usage:
    from odp.lib.metadata_adapters import adapt_metadata

    # Auto-detect and adapt
    normalized = adapt_metadata(raw_metadata)

    # Explicit schema
    normalized = adapt_metadata(raw_metadata, schema_id="SAEON.DataCite4")

    # With fallback
    normalized = adapt_metadata(raw_metadata, schema_id="auto", fallback=True)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

from odp.lib.metadata_pdf import (
    RecordMetadata,
    PersonInfo,
    GeographicExtent,
    TemporalExtent,
    License,
)


class MetadataAdapter(ABC):
    """Abstract base class for metadata schema adapters."""

    @abstractmethod
    def can_handle(self, metadata: Dict[str, Any]) -> bool:
        """Check if this adapter can handle the given metadata."""
        pass

    @abstractmethod
    def adapt(self, metadata: Dict[str, Any]) -> RecordMetadata:
        """Convert raw metadata to unified RecordMetadata format."""
        pass


class DataCiteAdapter(MetadataAdapter):
    """Adapter for DataCite 4 metadata schema.

    DataCite structure:
    {
        "titles": [{"title": "..."}],
        "doi": "10.15493/...",
        "publisher": "...",
        "publicationYear": 2024,
        "creators": [...],
        "descriptions": [...],
        "geoLocations": [...],
        "rightsList": [...],
        ...
    }
    """

    def can_handle(self, metadata: Dict[str, Any]) -> bool:
        """Check if this looks like DataCite metadata."""
        # DataCite has specific markers
        has_doi = "doi" in metadata
        has_titles = "titles" in metadata
        has_creators = "creators" in metadata
        return has_doi and has_titles and has_creators

    def adapt(self, metadata: Dict[str, Any]) -> RecordMetadata:
        """Convert DataCite metadata to unified format."""
        try:
            # Extract title
            title = "N/A"
            if "titles" in metadata and metadata["titles"]:
                title = metadata["titles"][0].get("title", "N/A")

            # Extract DOI
            doi = metadata.get("doi", "N/A")

            # Extract publisher and year
            publisher = metadata.get("publisher", "N/A")
            publication_year = str(metadata.get("publicationYear", "N/A"))

            # Extract abstract
            abstract = "N/A"
            if "descriptions" in metadata and metadata["descriptions"]:
                for desc in metadata["descriptions"]:
                    if desc.get("descriptionType") == "Abstract":
                        abstract = desc.get("description", "N/A")
                        break

            # Extract keywords
            keywords = metadata.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [keywords]
            keywords = [k for k in keywords if k]  # Filter empty strings

            # Extract creator
            creator = PersonInfo()
            if "creators" in metadata and metadata["creators"]:
                creator_data = metadata["creators"][0]
                creator.name = creator_data.get("name", "N/A")

                # Extract affiliation
                if "affiliation" in creator_data and creator_data["affiliation"]:
                    creator.affiliation = creator_data["affiliation"][0].get(
                        "affiliation", "N/A"
                    )

                # Extract email from affiliation string
                for aff in creator_data.get("affiliation", []):
                    aff_str = aff.get("affiliation", "")
                    if "email:" in aff_str:
                        creator.email = aff_str.split("email:")[-1].strip()
                        break

                # Extract ORCID
                for identifier in creator_data.get("nameIdentifiers", []):
                    if identifier.get("nameIdentifierScheme") == "ORCID":
                        creator.orcid = identifier.get("nameIdentifier", "N/A")
                        break

            # Extract contact/contributor
            contact = PersonInfo()
            if "contributors" in metadata and metadata["contributors"]:
                for contributor in metadata["contributors"]:
                    if contributor.get("contributorType") == "ContactPerson":
                        contact.name = contributor.get("name", "N/A")

                        if "affiliation" in contributor and contributor["affiliation"]:
                            contact.affiliation = contributor["affiliation"][0].get(
                                "affiliation", "N/A"
                            )

                        for aff in contributor.get("affiliation", []):
                            aff_str = aff.get("affiliation", "")
                            if "email:" in aff_str:
                                contact.email = aff_str.split("email:")[-1].strip()
                                break
                        break

            # Extract license
            license_info = License()
            if "rightsList" in metadata and metadata["rightsList"]:
                license_data = metadata["rightsList"][0]
                license_info.text = license_data.get("rights", "N/A")
                license_info.uri = license_data.get("rightsURI", "")

            # Extract geographic extent
            geography = None
            if "geoLocations" in metadata and metadata["geoLocations"]:
                geo_loc = metadata["geoLocations"][0]
                if "geoLocationBox" in geo_loc:
                    box = geo_loc["geoLocationBox"]
                    geography = GeographicExtent(
                        north=float(box.get("northBoundLatitude", 0)),
                        south=float(box.get("southBoundLatitude", 0)),
                        east=float(box.get("eastBoundLongitude", 0)),
                        west=float(box.get("westBoundLongitude", 0)),
                    )

            # Extract temporal extent (from record level, not metadata)
            temporal = TemporalExtent()

            return RecordMetadata(
                title=title,
                doi=doi,
                publisher=publisher,
                publication_year=publication_year,
                abstract=abstract,
                keywords=keywords,
                creator=creator,
                contact=contact,
                license=license_info,
                geography=geography,
                temporal=temporal,
            )

        except Exception as e:
            raise ValueError(f"DataCite adaptation failed: {str(e)}") from e


class ISO19115Adapter(MetadataAdapter):
    """Adapter for ISO 19115 metadata schema.

    ISO19115 structure:
    {
        "title": "...",
        "fileIdentifier": "...",
        "abstract": "...",
        "responsibleParties": [...],
        "constraints": [...],
        "extent": {...},
        ...
    }
    """

    def can_handle(self, metadata: Dict[str, Any]) -> bool:
        """Check if this looks like ISO19115 metadata."""
        # ISO19115 has specific markers
        has_title = "title" in metadata
        has_file_id = "fileIdentifier" in metadata
        has_responsible = "responsibleParties" in metadata
        return has_title and has_file_id and has_responsible

    def adapt(self, metadata: Dict[str, Any]) -> RecordMetadata:
        """Convert ISO19115 metadata to unified format."""
        try:
            # Extract title
            title = metadata.get("title", "N/A")

            # Extract DOI from fileIdentifier
            doi = metadata.get("fileIdentifier", "N/A")

            # Extract publisher from responsibleParties
            publisher = "N/A"
            if "responsibleParties" in metadata:
                for party in metadata["responsibleParties"]:
                    if party.get("role") == "publisher":
                        publisher = party.get("organizationName", "N/A")
                        break

            # Extract publication year (if available in metadata)
            publication_year = "N/A"

            # Extract abstract
            abstract = metadata.get("abstract", "N/A")

            # Extract keywords
            keywords = metadata.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [keywords]
            keywords = [k for k in keywords if k]

            # Extract creator from responsibleParties with role="originator"
            creator = PersonInfo()
            if "responsibleParties" in metadata:
                for party in metadata["responsibleParties"]:
                    if party.get("role") == "originator":
                        creator.name = party.get("individualName", "N/A")
                        creator.affiliation = party.get("organizationName", "N/A")

                        # Extract email from contactInfo
                        contact_info = party.get("contactInfo", "")
                        if "email:" in contact_info:
                            creator.email = contact_info.split("email:")[-1].strip()
                        break

            # Extract contact from responsibleParties with role="pointOfContact"
            contact = PersonInfo()
            if "responsibleParties" in metadata:
                for party in metadata["responsibleParties"]:
                    if party.get("role") == "pointOfContact":
                        contact.name = party.get("individualName", "N/A")
                        contact.affiliation = party.get("organizationName", "N/A")

                        contact_info = party.get("contactInfo", "")
                        if "email:" in contact_info:
                            contact.email = contact_info.split("email:")[-1].strip()
                        break

            # Extract license from constraints
            license_info = License()
            if "constraints" in metadata and metadata["constraints"]:
                constraint = metadata["constraints"][0]
                license_info.text = constraint.get("rights", "N/A")
                license_info.uri = constraint.get("rightsURI", "")

            # Extract geographic extent
            geography = None
            if "extent" in metadata:
                extent = metadata["extent"]
                if "geographicElements" in extent and extent["geographicElements"]:
                    geo_elem = extent["geographicElements"][0]
                    if "boundingBox" in geo_elem:
                        box = geo_elem["boundingBox"]
                        geography = GeographicExtent(
                            north=float(box.get("northBoundLatitude", 0)),
                            south=float(box.get("southBoundLatitude", 0)),
                            east=float(box.get("eastBoundLongitude", 0)),
                            west=float(box.get("westBoundLongitude", 0)),
                        )

            # Extract temporal extent
            temporal = TemporalExtent()

            return RecordMetadata(
                title=title,
                doi=doi,
                publisher=publisher,
                publication_year=publication_year,
                abstract=abstract,
                keywords=keywords,
                creator=creator,
                contact=contact,
                license=license_info,
                geography=geography,
                temporal=temporal,
            )

        except Exception as e:
            raise ValueError(f"ISO19115 adaptation failed: {str(e)}") from e


class AutoDetectAdapter(MetadataAdapter):
    """Adapter that auto-detects the schema and routes to appropriate adapter."""

    def __init__(self):
        """Initialize with available adapters."""
        self.adapters = [DataCiteAdapter(), ISO19115Adapter()]

    def can_handle(self, metadata: Dict[str, Any]) -> bool:
        """Auto-detect always claims to handle metadata."""
        return True

    def adapt(self, metadata: Dict[str, Any]) -> RecordMetadata:
        """Auto-detect schema and adapt using appropriate adapter."""
        # Try each adapter in order
        for adapter in self.adapters:
            try:
                if adapter.can_handle(metadata):
                    return adapter.adapt(metadata)
            except Exception:
                # Try next adapter
                continue

        # If no adapter matched, raise error
        raise ValueError(
            "Could not detect metadata schema. Ensure metadata matches DataCite4 or ISO19115 format."
        )


def adapt_metadata(
    raw_metadata: Dict[str, Any],
    schema_id: Optional[str] = None,
    fallback: bool = True,
) -> RecordMetadata:
    """Factory function to adapt metadata from various schemas.

    Args:
        raw_metadata: Raw metadata dictionary
        schema_id: Schema identifier:
            - "SAEON.DataCite4" or "datacite4": Use DataCite adapter
            - "SAEON.ISO19115" or "iso19115": Use ISO19115 adapter
            - "auto" or None: Auto-detect schema
        fallback: If True and schema_id doesn't match, try auto-detection

    Returns:
        RecordMetadata: Normalized metadata

    Raises:
        ValueError: If metadata cannot be adapted
    """
    # Map schema IDs to adapters
    adapter_map = {
        "datacite4": DataCiteAdapter(),
        "SAEON.DataCite4": DataCiteAdapter(),
        "iso19115": ISO19115Adapter(),
        "SAEON.ISO19115": ISO19115Adapter(),
        "auto": AutoDetectAdapter(),
        None: AutoDetectAdapter(),
    }

    # Get appropriate adapter
    adapter = adapter_map.get(schema_id)

    if adapter is None:
        # Unknown schema_id
        if fallback:
            # Fall back to auto-detection
            adapter = AutoDetectAdapter()
        else:
            raise ValueError(f"Unknown schema_id: {schema_id}")

    try:
        return adapter.adapt(raw_metadata)
    except Exception as e:
        if fallback and schema_id not in ("auto", None):
            # Try auto-detection as fallback
            try:
                return AutoDetectAdapter().adapt(raw_metadata)
            except Exception:
                pass
        raise e
