from datetime import datetime
from typing import Optional

from jschon import JSON, URI

from odp.api.models import PublishedMetadataModel, PublishedRecordModel, PublishedSAEONRecordModel, PublishedTagInstanceModel, RecordModel
from odp.catalog import Catalog
from odp.const import ODPCollectionTag, ODPMetadataSchema
from odp.const.db import SchemaType
from odp.db import Session
from odp.db.models import Schema
from odp.lib.schema import schema_catalog


class SAEONCatalog(Catalog):
    indexed = True

    def create_published_record(self, record_model: RecordModel) -> PublishedRecordModel:
        """Create the published form of a record."""
        return PublishedSAEONRecordModel(
            id=record_model.id,
            doi=record_model.doi,
            sid=record_model.sid,
            collection_key=record_model.collection_key,
            collection_name=record_model.collection_name,
            provider_key=record_model.provider_key,
            provider_name=record_model.provider_name,
            metadata_records=self._create_published_metadata(record_model),
            tags=self._create_published_tags(record_model),
            timestamp=record_model.timestamp,
        )

    @staticmethod
    def _create_published_metadata(record_model: RecordModel) -> list[PublishedMetadataModel]:
        """Create the published metadata outputs for a record."""
        published_metadata = [
            PublishedMetadataModel(
                schema_id=record_model.schema_id,
                schema_uri=record_model.schema_uri,
                metadata=record_model.metadata,
            )
        ]

        if record_model.schema_id == ODPMetadataSchema.SAEON_ISO19115:
            iso19115_schemaobj = Session.get(Schema, (ODPMetadataSchema.SAEON_ISO19115, SchemaType.metadata))
            datacite_schemaobj = Session.get(Schema, (ODPMetadataSchema.SAEON_DATACITE4, SchemaType.metadata))

            iso19115_jsonschema = schema_catalog.get_schema(URI(iso19115_schemaobj.uri))
            result = iso19115_jsonschema.evaluate(JSON(record_model.metadata))
            datacite_metadata = result.output(
                'translation',
                scheme='saeon/datacite4',
                ignore_validity=True,
                clear_empties=True,
            )

            published_metadata += [
                PublishedMetadataModel(
                    schema_id=datacite_schemaobj.id,
                    schema_uri=datacite_schemaobj.uri,
                    metadata=datacite_metadata,
                )
            ]

        return published_metadata

    @staticmethod
    def _create_published_tags(record_model: RecordModel) -> list[PublishedTagInstanceModel]:
        """Create the published tags for a record."""
        return [
            PublishedTagInstanceModel(
                tag_id=tag_instance.tag_id,
                data=tag_instance.data,
                user_name=tag_instance.user_name,
                timestamp=tag_instance.timestamp,
            ) for tag_instance in record_model.tags if tag_instance.public
        ]

    def create_text_index_data(
            self, published_record: PublishedSAEONRecordModel
    ) -> tuple[str, str]:
        """Create a title, free_text tuple from metadata field values to be indexed for full text search."""
        datacite_metadata = self._get_metadata_dict(published_record, ODPMetadataSchema.SAEON_DATACITE4)
        title = ''
        free_text = []

        for title in datacite_metadata.get('titles', ()):
            if title_text := title.get('title'):
                title = title_text

        if publisher := datacite_metadata.get('publisher'):
            free_text += [publisher]

        for creator in datacite_metadata.get('creators', ()):
            if creator_name := creator.get('name'):
                free_text += [creator_name]
            for affiliation in creator.get('affiliation', ()):
                if affiliation_text := affiliation.get('affiliation'):
                    free_text += [affiliation_text]

        for contributor in datacite_metadata.get('contributors', ()):
            if contributor_name := contributor.get('name'):
                free_text += [contributor_name]
            for affiliation in contributor.get('affiliation', ()):
                if affiliation_text := affiliation.get('affiliation'):
                    free_text += [affiliation_text]

        for subject in datacite_metadata.get('subjects', ()):
            if subject_text := subject.get('subject'):
                free_text += [subject_text]

        for description in datacite_metadata.get('descriptions', ()):
            if description_text := description.get('description'):
                free_text += [description_text]

        return (
            title,
            ' '.join(free_text),
        )

    def create_keyword_index_data(
            self, published_record: PublishedSAEONRecordModel
    ) -> list[str]:
        """Create an array of metadata keywords to be indexed for keyword search."""

        def _add_keyword(kw: str):
            nonlocal keyword_list, keyword_set
            if (kw := kw.strip()) and (kw_lower := kw.lower()) not in keyword_set:
                keyword_set |= {kw_lower}
                keyword_list += [kw]

        keyword_list = []
        keyword_set = set()

        if iso19115_metadata := self._get_metadata_dict(published_record, ODPMetadataSchema.SAEON_ISO19115):
            for keyword_obj in iso19115_metadata.get('descriptiveKeywords', ()):
                if keyword_obj.get('keywordType') in ('general', 'place', 'stratum'):
                    _add_keyword(keyword_obj.get('keyword', ''))

        else:
            datacite_metadata = self._get_metadata_dict(published_record, ODPMetadataSchema.SAEON_DATACITE4)
            for subject_obj in datacite_metadata.get('subjects', ()):
                _add_keyword(subject_obj.get('subject', ''))

        return sorted(keyword_list)

    def create_facet_index_data(
            self, published_record: PublishedSAEONRecordModel
    ) -> dict[str, list[str]]:
        """Create a mapping of facet names to values to be indexed for faceted search."""
        datacite_metadata = self._get_metadata_dict(published_record, ODPMetadataSchema.SAEON_DATACITE4)

        return {
            'Collection': [
                published_record.collection_name
            ],
            'Product': [
                tag.data['project']
                for tag in published_record.tags
                if tag.tag_id == ODPCollectionTag.PROJECT
            ],
            'License': [
                license
                for rights_obj in datacite_metadata.get('rightsList', ())
                if (license := rights_obj.get('rights'))
            ],
        }

    def create_spatial_index_data(
            self, published_record: PublishedSAEONRecordModel
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Create a N-E-S-W tuple of the spatial extent to be indexed for spatial search."""

        def _bump_lat(lat):
            nonlocal north, south
            if lat is not None:
                if north is None or lat > north:
                    north = lat
                if south is None or lat < south:
                    south = lat

        def _bump_lon(lon):
            nonlocal east, west
            if lon is not None:
                if east is None or lon > east:
                    east = lon
                if west is None or lon < west:
                    west = lon

        datacite_metadata = self._get_metadata_dict(published_record, ODPMetadataSchema.SAEON_DATACITE4)
        north = None
        east = None
        south = None
        west = None

        for geolocation in datacite_metadata.get('geoLocations', ()):
            if geobox := geolocation.get('geoLocationBox'):
                _bump_lat(geobox.get('northBoundLatitude'))
                _bump_lon(geobox.get('eastBoundLongitude'))
                _bump_lat(geobox.get('southBoundLatitude'))
                _bump_lon(geobox.get('westBoundLongitude'))

            for geopolygon in geolocation.get('geoLocationPolygons', ()):
                for geopoint in geopolygon.get('polygonPoints', ()):
                    _bump_lon(geopoint.get('pointLongitude'))
                    _bump_lat(geopoint.get('pointLatitude'))

            if geopoint := geolocation.get('geoLocationPoint'):
                _bump_lon(geopoint.get('pointLongitude'))
                _bump_lat(geopoint.get('pointLatitude'))

        return north, east, south, west

    def create_temporal_index_data(
            self, published_record: PublishedSAEONRecordModel
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Create a start-end tuple of the temporal extent to be indexed for temporal search.

        If the record has only a start date, it is taken to be its end date too.
        """

        def _get_dt(n):
            try:
                return datetime.fromisoformat(date_values[n])
            except (IndexError, ValueError):
                pass

        datacite_metadata = self._get_metadata_dict(published_record, ODPMetadataSchema.SAEON_DATACITE4)
        start = None
        end = None

        for date_obj in datacite_metadata.get('dates', ()):
            if date_obj.get('dateType') == 'Valid':
                if date_text := date_obj.get('date'):
                    date_values = date_text.split('/')
                    if start_dt := _get_dt(0):
                        if not start or start_dt < start:
                            start = start_dt
                    if end_dt := _get_dt(1):
                        if not end or end_dt > end:
                            end = end_dt

        if start and not end:
            end = start

        return start, end

    @staticmethod
    def _get_metadata_dict(
            published_record: PublishedSAEONRecordModel,
            schema_id: ODPMetadataSchema,
    ) -> Optional[dict]:
        return next((
            metadata_record.metadata
            for metadata_record in published_record.metadata_records
            if metadata_record.schema_id == schema_id
        ), None)
