from jschon import JSON, URI

from odp.api.models import PublishedDataCiteRecordModel, PublishedRecordModel, RecordModel
from odp.catalog import Catalog
from odp.config import config
from odp.const import DOI_PREFIX, ODPCatalog, ODPCollectionTag, ODPMetadataSchema
from odp.const.db import SchemaType
from odp.db import Session
from odp.db.models import CatalogRecord, Schema
from odp.lib.datacite import DataciteClient, DataciteRecordIn
from odp.lib.schema import schema_catalog


class DataCiteCatalog(Catalog):
    external = True

    def __init__(self, catalog_id: str) -> None:
        super().__init__(catalog_id)
        self.datacite = DataciteClient(
            api_url=config.DATACITE.API_URL,
            username=config.DATACITE.USERNAME,
            password=config.DATACITE.PASSWORD,
            doi_prefix=DOI_PREFIX,
        )
        self.catalog_api = config.ODP.API_URL + '/catalog'

    def evaluate_record(
            self,
            record_model: RecordModel,
            can_publish_reasons: list[str],
            cannot_publish_reasons: list[str],
    ) -> None:
        """Evaluate whether a record can be published.

        Only records with DOIs can be published to DataCite.
        """
        super().evaluate_record(record_model, can_publish_reasons, cannot_publish_reasons)

        if record_model.doi:
            can_publish_reasons += ['has DOI']
        else:
            cannot_publish_reasons += ['no DOI']

    def create_published_record(self, record_model: RecordModel) -> PublishedRecordModel:
        """Create the published form of a record."""
        if record_model.schema_id == ODPMetadataSchema.SAEON_DATACITE4:
            datacite_metadata = record_model.metadata

        elif record_model.schema_id == ODPMetadataSchema.SAEON_ISO19115:
            schema = Session.get(Schema, (record_model.schema_id, SchemaType.metadata))
            iso19115_schema = schema_catalog.get_schema(URI(schema.uri))
            result = iso19115_schema.evaluate(JSON(record_model.metadata))
            datacite_metadata = result.output('translation', scheme='saeon/datacite4', ignore_validity=True)

        elif record_model.schema_id == ODPMetadataSchema.SAEON_EML:
            schema = Session.get(Schema, (record_model.schema_id, SchemaType.metadata))
            eml_schema = schema_catalog.get_schema(URI(schema.uri))
            result = eml_schema.evaluate(JSON(record_model.metadata))
            datacite_metadata = result.output('translation', scheme='saeon/datacite4', ignore_validity=True)

        else:
            raise NotImplementedError

        return PublishedDataCiteRecordModel(
            doi=record_model.doi,
            url=self._doi_callback_url(record_model),
            metadata=datacite_metadata,
        )

    def _doi_callback_url(self, record_model: RecordModel) -> str:
        """Return the API URL for resolving a DOI to a catalog landing page."""
        tagged_mims = any((
            tag for tag in record_model.tags
            if tag.tag_id == ODPCollectionTag.INFRASTRUCTURE and tag.data['infrastructure'] == 'MIMS'
        ))

        catalog_id = ODPCatalog.MIMS if tagged_mims else ODPCatalog.SAEON
        return f'{self.catalog_api}/{catalog_id}/go/{record_model.doi}'

    def sync_external_record(self, record_id: str) -> None:
        """Create / update / delete a record on the DataCite platform."""
        catalog_record = Session.get(CatalogRecord, (self.catalog_id, record_id))
        if catalog_record.published:
            self.datacite.publish_doi(DataciteRecordIn(**catalog_record.published_record))
        elif doi := catalog_record.record.doi:
            self.datacite.unpublish_doi(doi)
