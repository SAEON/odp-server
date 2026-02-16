import re
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from jschon import JSON, JSONSchema
from sqlalchemy import func, select
from starlette.status import HTTP_403_FORBIDDEN, HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_ENTITY, HTTP_404_NOT_FOUND

from odp.api.lib.auth import Authorized
from odp.api.lib.utils import output_tag_instance_model
from odp.api.models import (
    RecordModel,
    RecordModelIn,
)
from odp.const import DOI_REGEX, ODPCollectionTag, ODPMetadataSchema
from odp.const.db import AuditCommand, SchemaType
from odp.db import Session
from odp.db.models import (
    CollectionTag,
    Record,
    RecordAudit, PublishedRecord, RecordTag, RecordTagAudit,
)


def create_record(
        record_in: RecordModelIn,
        metadata_schema: JSONSchema,
        auth: Authorized,
        ignore_collection_tags: bool = False,
) -> RecordModel:
    if auth.collection_ids != '*' and record_in.collection_id not in auth.collection_ids:
        raise HTTPException(HTTP_403_FORBIDDEN)

    if not ignore_collection_tags and Session.execute(
            select(CollectionTag).
                    where(CollectionTag.collection_id == record_in.collection_id).
                    where(CollectionTag.tag_id == ODPCollectionTag.FROZEN)
    ).first() is not None:
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'A record cannot be added to a frozen collection')

    if record_in.doi and Session.execute(
            select(Record).
                    where(func.lower(Record.doi) == record_in.doi.lower())
    ).first() is not None:
        raise HTTPException(HTTP_409_CONFLICT, 'DOI is already in use')

    if record_in.sid and Session.execute(
            select(Record).
                    where(Record.sid == record_in.sid)
    ).first() is not None:
        raise HTTPException(HTTP_409_CONFLICT, 'SID is already in use')

    record = Record(
        doi=record_in.doi,
        sid=record_in.sid,
        collection_id=record_in.collection_id,
        parent_id=get_parent_id(record_in.metadata, record_in.schema_id),
        schema_id=record_in.schema_id,
        schema_type=SchemaType.metadata,
        metadata_=record_in.metadata,
        validity=get_validity(record_in.metadata, metadata_schema),
        timestamp=(timestamp := datetime.now(timezone.utc)),
    )
    record.save()

    create_audit_record(auth, record, timestamp, AuditCommand.insert)

    touch_parent(record, timestamp)

    return output_record_model(record)


def get_parent_id(metadata: dict[str, Any], schema_id: ODPMetadataSchema) -> str | None:
    """Return the id of the parent record implied by an IsPartOf related identifier.

    The child-parent relationship is only established when both sides have a DOI.

    This is supported for the SAEON.DataCite4 and SAEON.ISO19115 metadata schemas.
    """
    try:
        child_doi = metadata['doi']
    except KeyError:
        return

    if schema_id not in (ODPMetadataSchema.SAEON_DATACITE4, ODPMetadataSchema.SAEON_ISO19115):
        return

    try:
        parent_refs = list(filter(
            lambda ref: ref['relationType'] == 'IsPartOf' and ref['relatedIdentifierType'] == 'DOI',
            metadata['relatedIdentifiers']
        ))
    except KeyError:
        return

    if not parent_refs:
        return

    if len(parent_refs) > 1:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY,
            'Cannot determine parent DOI: found multiple related identifiers with relation IsPartOf and type DOI.',
        )

    # related DOIs sometimes appear as doi.org links, sometimes as plain DOIs
    if match := re.search(DOI_REGEX[1:], parent_refs[0]['relatedIdentifier']):
        parent_doi = match.group(0)

        if parent_doi.lower() == child_doi.lower():
            raise HTTPException(
                HTTP_422_UNPROCESSABLE_ENTITY,
                'DOI cannot be a parent of itself.',
            )

        parent_record = Session.execute(
            select(Record).
            where(func.lower(Record.doi) == parent_doi.lower())
        ).scalar_one_or_none()

        if parent_record is None:
            raise HTTPException(
                HTTP_422_UNPROCESSABLE_ENTITY,
                f'Record not found for parent DOI {parent_doi}',
            )

    else:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY,
            'Parent reference is not a valid DOI.',
        )

    return parent_record.id


def create_audit_record(
        auth: Authorized,
        record: Record,
        timestamp: datetime,
        command: AuditCommand,
) -> None:
    RecordAudit(
        client_id=auth.client_id,
        user_id=auth.user_id,
        command=command,
        timestamp=timestamp,
        _id=record.id,
        _doi=record.doi,
        _sid=record.sid,
        _metadata=record.metadata_,
        _collection_id=record.collection_id,
        _schema_id=record.schema_id,
        _parent_id=record.parent_id,
    ).save()


def touch_parent(record: Record, timestamp: datetime) -> None:
    """Recursively update the timestamp of the given record's parent and
     its parent(s), where such exist."""
    if record.parent_id:
        # load parent record explicitly; record.parent might not be up-to-date at this point
        parent = Session.get(Record, record.parent_id)
        parent.timestamp = timestamp
        parent.save()
        touch_parent(parent, timestamp)


def output_record_model(record: Record) -> RecordModel:
    return RecordModel(
        id=record.id,
        doi=record.doi,
        sid=record.sid,
        collection_id=record.collection_id,
        collection_key=record.collection.key,
        collection_name=record.collection.name,
        provider_id=record.collection.provider_id,
        provider_key=record.collection.provider.key,
        provider_name=record.collection.provider.name,
        schema_id=record.schema_id,
        schema_uri=record.schema.uri,
        parent_id=record.parent_id,
        parent_doi=record.parent.doi if record.parent_id else None,
        child_dois={
            child.doi: child.id
            for child in record.children
        },
        metadata=record.metadata_,
        validity=record.validity,
        timestamp=record.timestamp.isoformat(),
        tags=[
                 output_tag_instance_model(collection_tag)
                 for collection_tag in record.collection.tags
             ] + [
                 output_tag_instance_model(record_tag)
                 for record_tag in record.tags
             ],
        published_catalog_ids=[
            catalog_record.catalog_id
            for catalog_record in record.catalog_records
            if catalog_record.published
        ]
    )


def get_validity(metadata: dict[str, Any], schema: JSONSchema) -> Any:
    if (result := schema.evaluate(JSON(metadata))).valid:
        return result.output('flag')

    return result.output('detailed')


def set_record(
        create: bool,
        record: Record,
        record_in: RecordModelIn,
        metadata_schema: JSONSchema,
        auth: Authorized,
        ignore_collection_tags: bool = False,
) -> RecordModel:
    if not create and auth.collection_ids != '*' and record.collection_id not in auth.collection_ids:
        raise HTTPException(HTTP_403_FORBIDDEN)

    if not ignore_collection_tags and Session.execute(
        select(CollectionTag).
        where(CollectionTag.collection_id == record_in.collection_id).
        where(CollectionTag.tag_id == ODPCollectionTag.FROZEN)
    ).first() is not None:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY,
            'Cannot update a record belonging to a frozen collection',
        )

    if record_in.doi and Session.execute(
        select(Record).
        where(Record.id != record.id).
        where(func.lower(Record.doi) == record_in.doi.lower())
    ).first() is not None:
        raise HTTPException(HTTP_409_CONFLICT, 'DOI is already in use')

    if record_in.sid and Session.execute(
        select(Record).
        where(Record.id != record.id).
        where(Record.sid == record_in.sid)
    ).first() is not None:
        raise HTTPException(HTTP_409_CONFLICT, 'SID is already in use')

    if record.doi is not None and record.doi != record_in.doi and Session.execute(
        select(PublishedRecord).
        where(PublishedRecord.doi == record.doi)
    ).first() is not None:
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'The DOI has been published and cannot be modified.')

    if (
        create or
        record.doi != record_in.doi or
        record.sid != record_in.sid or
        record.collection_id != record_in.collection_id or
        record.schema_id != record_in.schema_id or
        record.metadata_ != record_in.metadata
    ):
        record.doi = record_in.doi
        record.sid = record_in.sid
        record.collection_id = record_in.collection_id
        record.schema_id = record_in.schema_id
        record.schema_type = SchemaType.metadata
        record.metadata_ = record_in.metadata
        record.validity = get_validity(record_in.metadata, metadata_schema)
        record.timestamp = (timestamp := datetime.now(timezone.utc))

        parent_id = get_parent_id(record_in.metadata, record_in.schema_id)
        if record.parent_id != parent_id:
            touch_parent(record, timestamp)  # timestamp old parent for child removal
            record.parent_id = parent_id

        record.save()

        touch_parent(record, timestamp)

        create_audit_record(auth, record, timestamp, AuditCommand.insert if create else AuditCommand.update)

    return output_record_model(record)


def create_tag_audit_record(
        auth: Authorized,
        record_tag: RecordTag,
        timestamp: datetime,
        command: AuditCommand,
) -> None:
    RecordTagAudit(
        client_id=auth.client_id,
        user_id=auth.user_id,
        command=command,
        timestamp=timestamp,
        _id=record_tag.id,
        _record_id=record_tag.record_id,
        _tag_id=record_tag.tag_id,
        _user_id=record_tag.user_id,
        _data=record_tag.data,
    ).save()


def delete_record_(
        record_id: str,
        auth: Authorized,
        ignore_collection_tags: bool = False,
) -> None:
    if not (record := Session.get(Record, record_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if auth.collection_ids != '*' and record.collection_id not in auth.collection_ids:
        raise HTTPException(HTTP_403_FORBIDDEN)

    if not ignore_collection_tags and Session.execute(
            select(CollectionTag).
                    where(CollectionTag.collection_id == record.collection_id).
                    where(CollectionTag.tag_id == ODPCollectionTag.FROZEN)
    ).first() is not None:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY,
            'Cannot delete a record belonging to a frozen collection',
        )

    if Session.get(PublishedRecord, record_id):
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY,
            'The record has been published and cannot be deleted. Please retract the record instead.',
        )

    touch_parent(record, timestamp := datetime.now(timezone.utc))

    create_audit_record(auth, record, timestamp, AuditCommand.delete)

    record.delete()


def untag_record_(
        record_id: str,
        tag_instance_id: str,
        auth: Authorized,
        ignore_user_id: bool = False,
) -> None:
    if not (record := Session.get(Record, record_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if auth.collection_ids != '*' and record.collection_id not in auth.collection_ids:
        raise HTTPException(HTTP_403_FORBIDDEN)

    if not (record_tag := Session.execute(
            select(RecordTag).
                    where(RecordTag.id == tag_instance_id).
                    where(RecordTag.record_id == record_id)
    ).scalar_one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if not ignore_user_id and record_tag.user_id != auth.user_id:
        raise HTTPException(HTTP_403_FORBIDDEN)

    record_tag.delete()

    record.timestamp = (timestamp := datetime.now(timezone.utc))
    record.save()

    touch_parent(record, timestamp)

    create_tag_audit_record(auth, record_tag, timestamp, AuditCommand.delete)