import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from odp.lib.pdf_generator import (
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

    def _parse_person_info(self, person_data: Dict[str, Any]) -> PersonInfo:
        """
        Shared helper to extract person details and reduce code duplication.
        Handles varied structures between DataCite and ISO19115.
        """
        person = PersonInfo()
        person.name = (
                person_data.get("name") or
                person_data.get("individualName") or
                "N/A"
        )

        # Extract affiliation / organization
        affiliations = person_data.get("affiliation", [])
        if isinstance(affiliations, list) and affiliations:
            person.affiliation = affiliations[0].get("affiliation", "N/A")
        else:
            person.affiliation = person_data.get("organizationName", "N/A")

        # Extract email from affiliation strings or contactInfo
        search_target = str(person_data.get("contactInfo", ""))
        if not search_target and isinstance(affiliations, list):
            search_target = " ".join([str(a.get("affiliation", "")) for a in affiliations])

        if "email:" in search_target.lower():
            # Basic extraction: split at email: and take the next word
            parts = search_target.lower().split("email:")
            if len(parts) > 1:
                person.email = parts[-1].strip().split()[0].rstrip(',;')
            # Strip the email portion from affiliation so it isn't rendered twice
            if person.email and person.email != "N/A":
                person.affiliation = re.sub(
                    r',?\s*email:\s*\S+', '', person.affiliation, flags=re.IGNORECASE
                ).strip().rstrip(',').strip()

        # Extract ORCID (DataCite specific)
        for identifier in person_data.get("nameIdentifiers", []):
            if identifier.get("nameIdentifierScheme") == "ORCID":
                person.orcid = identifier.get("nameIdentifier", "N/A")

        return person


class DataCiteAdapter(MetadataAdapter):
    """Adapter for DataCite 4 metadata schema."""

    def can_handle(self, metadata: Dict[str, Any]) -> bool:
        return "doi" in metadata and "titles" in metadata and "creators" in metadata

    def adapt(self, metadata: Dict[str, Any]) -> RecordMetadata:
        # Use helper for creators
        creator = PersonInfo()
        if metadata.get("creators"):
            creator = self._parse_person_info(metadata["creators"][0])

        # Use helper for contact person
        contact = PersonInfo()
        for contributor in metadata.get("contributors", []):
            if contributor.get("contributorType") == "ContactPerson":
                contact = self._parse_person_info(contributor)
                break

        # Extract abstract
        abstract = "N/A"
        for desc in metadata.get("descriptions", []):
            if desc.get("descriptionType") == "Abstract":
                abstract = desc.get("description", "N/A")
                break

        # Extract license
        license_info = License()
        if metadata.get("rightsList"):
            rights = metadata["rightsList"][0]
            license_info.text = rights.get("rights", "N/A")
            license_info.uri = rights.get("rightsURI", "")

        # Extract geography
        geography = None
        if metadata.get("geoLocations"):
            box = metadata["geoLocations"][0].get("geoLocationBox")
            coord_keys = {"northBoundLatitude", "southBoundLatitude", "eastBoundLongitude", "westBoundLongitude"}
            if box and coord_keys.intersection(box.keys()):
                geography = GeographicExtent(
                    north=float(box.get("northBoundLatitude", 0)),
                    south=float(box.get("southBoundLatitude", 0)),
                    east=float(box.get("eastBoundLongitude", 0)),
                    west=float(box.get("westBoundLongitude", 0)),
                )
        subjects = metadata.get("subjects", [])
        keywords = [s.get("subject") for s in subjects if isinstance(s, dict) and s.get("subject")]
        if not keywords:
            keywords = metadata.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        keywords = [k for k in keywords if k]

        # Extract temporal extent from dates[].dateType == 'Valid'
        temporal = TemporalExtent()
        for date_obj in metadata.get("dates", []):
            if date_obj.get("dateType") == "Valid":
                if date_text := date_obj.get("date"):
                    parts = date_text.split("/")
                    temporal.start_date = parts[0].split("T")[0]
                    temporal.end_date = parts[1].split("T")[0] if len(parts) > 1 else parts[0].split("T")[0]
                break

        return RecordMetadata(
            title=metadata["titles"][0].get("title", "N/A") if metadata.get("titles") else "N/A",
            doi=metadata.get("doi", "N/A"),
            publisher=metadata.get("publisher", "N/A"),
            publication_year=str(metadata.get("publicationYear", "N/A")),
            abstract=abstract,
            keywords=keywords,
            creator=creator,
            contact=contact,
            license=license_info,
            geography=geography,
            temporal=temporal,
        )


class ISO19115Adapter(MetadataAdapter):
    """Adapter for ISO 19115 metadata schema."""

    def can_handle(self, metadata: Dict[str, Any]) -> bool:
        return "title" in metadata and "fileIdentifier" in metadata and "responsibleParties" in metadata

    def adapt(self, metadata: Dict[str, Any]) -> RecordMetadata:
        creator = PersonInfo()
        contact = PersonInfo()
        publisher = "N/A"

        for party in metadata.get("responsibleParties", []):
            role = party.get("role")
            if role == "originator":
                creator = self._parse_person_info(party)
            elif role == "pointOfContact":
                contact = self._parse_person_info(party)
            elif role == "publisher":
                publisher = party.get("organizationName", "N/A")

        # Extract license
        license_info = License()
        if metadata.get("constraints"):
            constraint = metadata["constraints"][0]
            license_info.text = constraint.get("rights", "N/A")
            license_info.uri = constraint.get("rightsURI", "")

        # Extract geography
        geography = GeographicExtent(north=0, south=0, east=0, west=0)
        if metadata.get("extent") and metadata["extent"].get("geographicElements"):
            box = metadata["extent"]["geographicElements"][0].get("boundingBox")
            if box:
                geography = GeographicExtent(
                    north=float(box.get("northBoundLatitude", 0)),
                    south=float(box.get("southBoundLatitude", 0)),
                    east=float(box.get("eastBoundLongitude", 0)),
                    west=float(box.get("westBoundLongitude", 0)),
                )

        # Extract temporal extent from extent.temporalElement
        temporal = TemporalExtent()
        temporal_element = metadata.get("extent", {}).get("temporalElement", {})
        if temporal_element:
            if start := temporal_element.get("startTime", ""):
                temporal.start_date = start.split("T")[0]
            if end := temporal_element.get("endTime", ""):
                temporal.end_date = end.split("T")[0]

        return RecordMetadata(
            title=metadata.get("title", "N/A"),
            doi=metadata.get("fileIdentifier", "N/A"),
            publisher=publisher,
            publication_year="N/A",
            abstract=metadata.get("abstract", "N/A"),
            keywords=_normalise_keywords(metadata.get("keywords", [])),
            creator=creator,
            contact=contact,
            license=license_info,
            geography=geography,
            temporal=temporal,
        )


def _normalise_keywords(keywords):
    """Convert keywords to a filtered list of non-empty strings."""
    if isinstance(keywords, str):
        keywords = [keywords]
    return [k for k in keywords if k]


def adapt_metadata(
        raw_metadata: Dict[str, Any],
        schema_id: Optional[str] = None,
        fallback: bool = True,
) -> RecordMetadata:
    """
    Factory function to adapt metadata from various schemas.

    If schema_id is recognised, that adapter is tried first. When fallback=True
    and the explicit adapter fails (or schema_id is unknown), auto-detection is
    attempted. When fallback=False, any failure raises ValueError immediately.
    """
    known = {
        "SAEON.DataCite4": DataCiteAdapter,
        "datacite4": DataCiteAdapter,
        "SAEON.ISO19115": ISO19115Adapter,
        "iso19115": ISO19115Adapter,
    }

    if schema_id is not None and schema_id not in known:
        if not fallback:
            raise ValueError(f"Unknown schema_id: {schema_id!r}")
        # fall through to auto-detection below
    elif schema_id in known:
        adapter = known[schema_id]()
        if adapter.can_handle(raw_metadata):
            try:
                return adapter.adapt(raw_metadata)
            except Exception:
                if not fallback:
                    raise ValueError(f"Could not adapt metadata with schema {schema_id!r}.")
        else:
            if not fallback:
                raise ValueError(f"Could not adapt metadata with schema {schema_id!r}.")
            # fall through to auto-detection

    # Auto-detection
    adapters = [DataCiteAdapter(), ISO19115Adapter()]
    for adapter in adapters:
        if adapter.can_handle(raw_metadata):
            try:
                return adapter.adapt(raw_metadata)
            except Exception:
                continue

    raise ValueError("Could not detect or adapt metadata schema. Supported: DataCite4, ISO19115.")
