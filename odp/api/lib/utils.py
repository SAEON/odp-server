from typing import Optional

from odp.api.models import PublishedDataCiteRecordModel, PublishedRecordModel, PublishedSAEONRecordModel, \
    TagInstanceModel
from odp.const import ODPCatalog
from odp.db.models import CatalogRecord, CollectionTag, RecordTag


def output_tag_instance_model(tag_instance: CollectionTag | RecordTag) -> TagInstanceModel:
    return TagInstanceModel(
        id=tag_instance.id,
        tag_id=tag_instance.tag_id,
        user_id=tag_instance.user_id,
        user_name=tag_instance.user.name if tag_instance.user_id else None,
        data=tag_instance.data,
        timestamp=tag_instance.timestamp.isoformat(),
        cardinality=tag_instance.tag.cardinality,
        public=tag_instance.tag.public,
    )


def output_published_record_model(catalog_record: CatalogRecord) -> Optional[PublishedRecordModel]:
    if not catalog_record.published:
        return None

    if catalog_record.catalog_id in (ODPCatalog.SAEON, ODPCatalog.MIMS):
        return PublishedSAEONRecordModel(**catalog_record.published_record | dict(
            keywords=catalog_record.keywords,
            spatial_north=catalog_record.spatial_north,
            spatial_east=catalog_record.spatial_east,
            spatial_south=catalog_record.spatial_south,
            spatial_west=catalog_record.spatial_west,
            temporal_start=catalog_record.temporal_start.isoformat() if catalog_record.temporal_start else None,
            temporal_end=catalog_record.temporal_end.isoformat() if catalog_record.temporal_end else None,
            searchable=catalog_record.searchable,
        ))

    if catalog_record.catalog_id == ODPCatalog.DATACITE:
        return PublishedDataCiteRecordModel(**catalog_record.published_record)


def remove_empty_elements(data):
    """
    Recursively removes None, empty strings, empty lists, and empty dicts.
    """
    if isinstance(data, dict):
        return {
            k: v for k, v in ((k, remove_empty_elements(v)) for k, v in data.items())
            if v not in (None, [], {}, "")
        }
    elif isinstance(data, list):
        return [
            v for v in (remove_empty_elements(v) for v in data)
            if v not in (None, [], {}, "")
        ]
    else:
        return data
