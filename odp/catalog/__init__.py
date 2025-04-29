import logging
from datetime import date, datetime, timezone
from typing import Any, Optional, final

from sqlalchemy import func, or_, select

from odp.api.lib.utils import output_published_record_model
from odp.api.models import PublishedRecordModel, RecordModel
from odp.api.routers.record import output_record_model
from odp.const import ODPCatalog, ODPCollectionTag, ODPMetadataSchema, ODPRecordTag
from odp.db import Session
from odp.db.models import Catalog as CatalogORM, CatalogRecord, CatalogRecordFacet, Collection, Provider, \
    PublishedRecord, Record, RecordTag

logger = logging.getLogger(__name__)


class Catalog:
    indexed = False
    """Whether to save indexing data to catalog records.
    If true, the following methods must be implemented:

    * create_text_index_data
    * create_keyword_index_data
    * create_facet_index_data
    * create_spatial_index_data
    * create_temporal_index_data
    """

    external = False
    """Whether to synchronize catalog records with an external system.
    If true, method sync_external_record must be implemented.
    """

    max_attempts = 3
    """Maximum number of consecutive attempts at externally synchronizing a record."""

    def __init__(self, catalog_id: str) -> None:
        self.catalog_id = catalog_id
        self.snapshot: dict[str, tuple[RecordModel, datetime]] = {}
        """Mapping of record UUIDs to tuples of (record_model, timestamp)."""

    @final
    def publish(self) -> None:
        """Fully update the public state of a catalog."""
        records = self._select_records()
        logger.info(f'{self.catalog_id} catalog: {(total := len(records))} records selected for evaluation')

        if total:
            self._create_snapshot(records)
            published = self._sync_catalog()
            logger.info(f'{self.catalog_id} catalog: {published} records published; {total - published} records hidden')

        if self.external:
            self._sync_external()

    def _select_records(self) -> list[tuple[str, datetime]]:
        """Select records to be evaluated for publication to, or
        retraction from, a catalog.

        A record is selected if:

        * there is no corresponding catalog_record entry; or
        * the record has any embargo tags; or
        * catalog_record.timestamp is less than any of:

          * collection.timestamp
          * provider.timestamp
          * record.timestamp

        :return: a list of (record_id, timestamp) tuples, where
            timestamp is that of the latest contributing change
        """
        records_subq = (
            select(
                Record.id.label('record_id'),
                func.greatest(
                    Collection.timestamp,
                    Provider.timestamp,
                    Record.timestamp,
                ).label('max_timestamp')
            ).
            join(Collection).
            join(Provider).
            subquery()
        )

        catalog_records_subq = (
            select(
                CatalogRecord.record_id,
                CatalogRecord.timestamp
            ).
            where(CatalogRecord.catalog_id == self.catalog_id).
            subquery()
        )

        stmt = (
            select(
                records_subq.c.record_id,
                records_subq.c.max_timestamp
            ).
            outerjoin_from(records_subq, catalog_records_subq).
            where(or_(
                catalog_records_subq.c.record_id == None,
                catalog_records_subq.c.timestamp < records_subq.c.max_timestamp,
                catalog_records_subq.c.record_id.in_(
                    select(RecordTag.record_id).
                    where(RecordTag.tag_id == ODPRecordTag.EMBARGO)
                )
            ))
        )

        return Session.execute(stmt).all()

    def _create_snapshot(self, records: list[tuple[str, datetime]]) -> None:
        """Create a snapshot of API record output models for the selected
        records.

        To ensure consistency of lookup info across all catalog records,
        the DB engine should be created with a transaction isolation level
        of 'REPEATABLE READ'.
        """
        logger.debug(f'{self.catalog_id} catalog: Creating snapshot...')
        for record_id, timestamp in records:
            record = Session.get(Record, record_id)
            record_model = output_record_model(record)
            self.snapshot[record_id] = (record_model, timestamp)

    def _sync_catalog(self) -> int:
        """Update the catalog from the snapshot, and return the number of
        records published."""
        logger.debug(f'{self.catalog_id} catalog: Synchronizing catalog records...')

        published = 0
        for record_id in self.snapshot:
            published += self._sync_catalog_record(record_id)

        catalog = Session.get(CatalogORM, self.catalog_id)
        catalog.data = self.create_global_data()
        catalog.timestamp = datetime.now(timezone.utc)
        catalog.save()

        Session.commit()

        return published

    def _sync_catalog_record(self, record_id: str) -> bool:
        """Synchronize a catalog_record entry with the current state of the
        corresponding record.

        The catalog_record entry is stamped with the `timestamp` of the latest
        contributing change (from record / collection / provider).
        """
        catalog_record = (Session.get(CatalogRecord, (self.catalog_id, record_id)) or
                          CatalogRecord(catalog_id=self.catalog_id, record_id=record_id))

        record_model, timestamp = self.snapshot[record_id]

        can_publish_reasons = []
        cannot_publish_reasons = []
        self.evaluate_record(record_model, can_publish_reasons, cannot_publish_reasons)

        if not cannot_publish_reasons:
            self._save_published_record(record_model)
            self._process_embargoes(record_model)
            catalog_record.published = True
            catalog_record.published_record = self.create_published_record(record_model).dict()
            catalog_record.reason = ' | '.join(can_publish_reasons)
        else:
            catalog_record.published = False
            catalog_record.published_record = None
            catalog_record.reason = ' | '.join(cannot_publish_reasons)

        catalog_record.timestamp = timestamp
        catalog_record.save()

        if self.indexed:
            self._index_catalog_record(catalog_record)
            catalog_record.save()

        if self.external:
            catalog_record.synced = False
            catalog_record.error = None
            catalog_record.error_count = 0
            catalog_record.save()

        Session.commit()

        return catalog_record.published

    def evaluate_record(
            self,
            record_model: RecordModel,
            can_publish_reasons: list[str],
            cannot_publish_reasons: list[str],
    ) -> None:
        """Evaluate whether a record can be published.

        Universal rules are defined here; derived Catalog classes
        may extend these with catalog-specific rules.

        Reasons for publishing MAY be added to can_publish_reasons.

        Reasons for not publishing MUST be added to cannot_publish_reasons:
        the caller uses this to decide whether or not to publish the record.
        """
        # tag for a record migrated without any subsequent changes
        migrated_tag = next(
            (tag for tag in record_model.tags if tag.tag_id == ODPRecordTag.MIGRATED and
             datetime.fromisoformat(tag.timestamp) >= datetime.fromisoformat(record_model.timestamp)),
            None
        )
        collection_published = any(
            (tag for tag in record_model.tags if tag.tag_id == ODPCollectionTag.PUBLISHED)
        )
        collection_harvested = any(
            (tag for tag in record_model.tags if tag.tag_id == ODPCollectionTag.HARVESTED)
        )
        qc_passed = any(
            (tag for tag in record_model.tags if tag.tag_id == ODPRecordTag.QC and tag.data['pass_'])
        ) and not any(
            (tag for tag in record_model.tags if tag.tag_id == ODPRecordTag.QC and not tag.data['pass_'])
        )
        retracted = any(
            (tag for tag in record_model.tags if tag.tag_id == ODPRecordTag.RETRACTED)
        )
        metadata_valid = record_model.validity['valid']

        # collection published tag is required in all cases
        if collection_published:
            can_publish_reasons += ['collection published']
        else:
            cannot_publish_reasons += ['collection not published']

        if migrated_tag:
            if migrated_tag.data['published']:
                can_publish_reasons += ['migrated as published']
            else:
                cannot_publish_reasons += ['migrated as not published']

        else:
            # for harvested collections, metadata must be valid, QC is ignored
            if collection_harvested:
                can_publish_reasons += ['collection harvested']
                if not metadata_valid:
                    cannot_publish_reasons += ['metadata invalid']

            # for non-harvested collections, QC is checked, metadata validity is ignored
            elif qc_passed:
                can_publish_reasons += ['QC passed']

            else:
                cannot_publish_reasons += ['QC failed']

            if retracted:
                cannot_publish_reasons += ['record retracted']

    def create_published_record(self, record_model: RecordModel) -> PublishedRecordModel:
        """Create the published form of a record."""
        raise NotImplementedError

    @staticmethod
    def _save_published_record(record_model: RecordModel) -> None:
        """Permanently save the record id and DOI when first published."""
        if not (published_record := Session.get(PublishedRecord, record_model.id)):
            published_record = PublishedRecord(id=record_model.id, doi=record_model.doi)
        elif record_model.doi and not published_record.doi:
            published_record.doi = record_model.doi
        else:
            return

        published_record.save()

    @staticmethod
    def _process_embargoes(record_model: RecordModel) -> None:
        """Check if a record is currently subject to an embargo and, if so, update
        the given `record_model`, stripping out download links / embedded datasets
        from the metadata."""
        current_date = date.today()
        embargoed = False

        for tag in record_model.tags:
            if tag.tag_id == ODPRecordTag.EMBARGO:
                start_date = date.fromisoformat(tag.data['start'])
                end_date = date.fromisoformat(tag.data['end'] or '3000-01-01')
                if start_date <= current_date <= end_date:
                    embargoed = True
                    break

        if not embargoed:
            return

        if record_model.schema_id == ODPMetadataSchema.SAEON_DATACITE4:
            try:
                if 'resourceDownload' in record_model.metadata['immutableResource']:
                    record_model.metadata['immutableResource']['resourceDownload']['downloadURL'] = None
            except KeyError:
                pass
            try:
                if 'resourceData' in record_model.metadata['immutableResource']:
                    record_model.metadata['immutableResource']['resourceData'] = None
            except KeyError:
                pass

        elif record_model.schema_id == ODPMetadataSchema.SAEON_ISO19115:
            for item in record_model.metadata.get('onlineResources', []):
                try:
                    if item['description'] == 'download':
                        item['linkage'] = None
                except KeyError:
                    pass

    def _sync_external(self) -> None:
        """Synchronize with an external catalog."""
        unsynced_catalog_records = Session.execute(
            select(CatalogRecord).
            where(CatalogRecord.catalog_id == self.catalog_id).
            where(CatalogRecord.synced == False).
            where(CatalogRecord.error_count < self.max_attempts)
        ).scalars().all()

        logger.info(
            f'{self.catalog_id} catalog: {(total := len(unsynced_catalog_records))} records selected for external sync')
        synced = 0

        for catalog_record in unsynced_catalog_records:
            try:
                self.sync_external_record(catalog_record.record_id)
                catalog_record.synced = True
                catalog_record.error = None
                catalog_record.error_count = 0
                synced += 1
            except Exception as e:
                catalog_record.error = repr(e)
                catalog_record.error_count += 1

            catalog_record.save()
            Session.commit()

        if total:
            logger.info(f'{self.catalog_id} catalog: {synced} records synced; {total - synced} errors')

    def sync_external_record(self, record_id: str) -> None:
        """Create / update / delete a record on an external catalog."""
        raise NotImplementedError

    def _index_catalog_record(self, catalog_record: CatalogRecord) -> None:
        """Compute and store search data for a catalog record."""
        if catalog_record.published:
            published_record = output_published_record_model(catalog_record)

            title, free_text = self.create_text_index_data(published_record)

            keywords = self.create_keyword_index_data(published_record)

            catalog_record.full_text = (
                (func.setweight(func.to_tsvector('english', ' '.join(keywords)), 'A'))
                .op('||')
                (func.setweight(func.to_tsvector('english', title), 'B'))
                .op('||')
                (func.setweight(func.to_tsvector('english', free_text), 'C'))
            )

            catalog_record.keywords = keywords

            catalog_record.facets = []
            for facet_name, facet_values in self.create_facet_index_data(published_record).items():
                for facet_value in facet_values:
                    catalog_record.facets += [CatalogRecordFacet(
                        catalog_id=catalog_record.catalog_id,
                        record_id=catalog_record.record_id,
                        facet=facet_name,
                        value=facet_value,
                    )]

            (catalog_record.spatial_north,
             catalog_record.spatial_east,
             catalog_record.spatial_south,
             catalog_record.spatial_west) = self.create_spatial_index_data(published_record)

            (catalog_record.temporal_start,
             catalog_record.temporal_end) = self.create_temporal_index_data(published_record)

            # strictly speaking, reading tag instances directly from the DB bypasses the
            # record snapshot, but this avoids having to publish the not-searchable tags
            catalog_record.searchable = not any(
                tag for tag in catalog_record.record.tags if tag.tag_id == ODPRecordTag.NOTSEARCHABLE
            ) and not any(
                tag for tag in catalog_record.record.collection.tags if tag.tag_id == ODPCollectionTag.NOTSEARCHABLE
            )

        else:
            catalog_record.full_text = None
            catalog_record.keywords = None
            catalog_record.facets = []
            catalog_record.spatial_north = None
            catalog_record.spatial_east = None
            catalog_record.spatial_south = None
            catalog_record.spatial_west = None
            catalog_record.temporal_start = None
            catalog_record.temporal_end = None
            catalog_record.searchable = None

    def create_text_index_data(
            self, published_record: PublishedRecordModel
    ) -> tuple[str, str]:
        """Create a title, free_text tuple from metadata field values to be indexed for full text search."""
        raise NotImplementedError

    def create_keyword_index_data(
            self, published_record: PublishedRecordModel
    ) -> list[str]:
        """Create an array of metadata keywords to be indexed for keyword search."""
        raise NotImplementedError

    def create_facet_index_data(
            self, published_record: PublishedRecordModel
    ) -> dict[str, list[str]]:
        """Create a mapping of facet names to values to be indexed for faceted search."""
        raise NotImplementedError

    def create_spatial_index_data(
            self, published_record: PublishedRecordModel
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Create a N-E-S-W tuple of the spatial extent to be indexed for spatial search."""
        raise NotImplementedError

    def create_temporal_index_data(
            self, published_record: PublishedRecordModel
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Create a start-end tuple of the temporal extent to be indexed for temporal search."""
        raise NotImplementedError

    def create_global_data(self) -> Any:
        """Create a JSON-compatible object to be published as global data for the catalog."""
        return None


def publish_all():
    from odp.catalog.datacite import DataCiteCatalog
    from odp.catalog.mims import MIMSCatalog
    from odp.catalog.saeon import SAEONCatalog

    catalog_classes = {
        ODPCatalog.SAEON: SAEONCatalog,
        ODPCatalog.MIMS: MIMSCatalog,
        ODPCatalog.DATACITE: DataCiteCatalog,
    }

    logger.info('PUBLISHING STARTED')
    try:
        for catalog_id, catalog_cls in catalog_classes.items():
            catalog_cls(catalog_id).publish()

        logger.info('PUBLISHING FINISHED')

    except Exception as e:
        logger.critical(f'PUBLISHING ABORTED: {str(e)}')
        raise
