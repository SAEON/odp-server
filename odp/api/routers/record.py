from datetime import datetime, timezone
from functools import partial
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from jschon import JSON, JSONSchema
from pydantic import constr
from sqlalchemy import and_, func, literal_column, null, or_, select, union_all
from sqlalchemy.orm import aliased
from starlette.status import HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND, HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_ENTITY

from odp.api.lib.auth import Authorize, Authorized, TagAuthorize, UntagAuthorize
from odp.api.lib.paging import Page, Paginator
from odp.api.lib.record import create_new_record, set_record, output_record_model, create_tag_audit_record, touch_parent, \
    delete_record_, untag_record_
from odp.api.lib.schema import get_metadata_schema, get_tag_schema
from odp.api.lib.utils import output_published_record_model, output_tag_instance_model
from odp.api.models import (
    AuditModel,
    CatalogRecordModel,
    RecordAuditModel,
    RecordModel,
    RecordModelIn,
    RecordTagAuditModel,
    TagInstanceModel,
    TagInstanceModelIn,
)
from odp.const import DOI_REGEX, ODPMetadataSchema, ODPScope
from odp.const.db import AuditCommand, TagCardinality, TagType
from odp.db import Session
from odp.db.models import (
    CatalogRecord,
    Collection,
    Record,
    RecordAudit,
    RecordTag,
    RecordTagAudit,
    Tag,
    VocabularyTerm,
)

router = APIRouter()


def output_catalog_record_model(catalog_record: CatalogRecord) -> CatalogRecordModel:
    return CatalogRecordModel(
        catalog_id=catalog_record.catalog_id,
        record_id=catalog_record.record_id,
        published=catalog_record.published,
        published_record=output_published_record_model(catalog_record),
        reason=catalog_record.reason,
        timestamp=catalog_record.timestamp.isoformat(),
        external_synced=catalog_record.synced,
        external_error=catalog_record.error,
        external_error_count=catalog_record.error_count,
        index_full_text=catalog_record.full_text,
        index_keywords=catalog_record.keywords,
        index_facets=[{'facet': facet.facet, 'value': facet.value}
                      for facet in catalog_record.facets],
        index_spatial_north=catalog_record.spatial_north,
        index_spatial_east=catalog_record.spatial_east,
        index_spatial_south=catalog_record.spatial_south,
        index_spatial_west=catalog_record.spatial_west,
        index_temporal_start=catalog_record.temporal_start.isoformat() if catalog_record.temporal_start else None,
        index_temporal_end=catalog_record.temporal_end.isoformat() if catalog_record.temporal_end else None,
        index_searchable=catalog_record.searchable,
    )


@router.get(
    '/',
    response_model=Page[RecordModel],
)
async def list_records(
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_READ)),
        paginator: Paginator = Depends(),
        collection_id: list[str] = Query(None),
        parent_id: str = None,
        identifier_q: str = None,
        title_q: str = None,
):
    stmt = (
        select(Record).
        join(Collection)
    )

    if collection_id:
        stmt = stmt.where(Collection.id.in_(collection_id))

    if parent_id:
        stmt = stmt.where(Record.parent_id == parent_id)

    if identifier_q and (id_terms := identifier_q.split()):
        id_exprs = []
        for id_term in id_terms:
            id_exprs += [
                Record.id.ilike(f'%{id_term}%'),
                Record.doi.ilike(f'%{id_term}%'),
                Record.sid.ilike(f'%{id_term}%'),
            ]
        stmt = stmt.where(or_(*id_exprs))

    if title_q and (title_terms := title_q.split()):
        datacite_title_exprs = [
            Record.metadata_['titles'][0]['title'].astext.ilike(f'%{title_term}%')
            for title_term in title_terms
        ]
        iso19115_title_exprs = [
            Record.metadata_['title'].astext.ilike(f'%{title_term}%')
            for title_term in title_terms
        ]
        stmt = stmt.where(or_(
            and_(
                Record.schema_id == ODPMetadataSchema.SAEON_DATACITE4,
                *datacite_title_exprs,
            ),
            and_(
                Record.schema_id == ODPMetadataSchema.SAEON_ISO19115,
                *iso19115_title_exprs,
            ),
        ))

    return paginator.paginate(
        stmt,
        lambda row: output_record_model(row.Record),
        sort='collection.key, record.doi, record.sid',
    )


@router.get(
    '/{record_id}',
    response_model=RecordModel,
)
async def get_record(
        record_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_READ)),
):
    if not (record := Session.get(Record, record_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_record_model(record)


@router.get(
    '/doi/{record_doi:path}',
    response_model=RecordModel,
)
async def get_record_by_doi(
        record_doi: constr(regex=DOI_REGEX),
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_READ)),
):
    if not (record := Session.execute(
            select(Record).
                    where(func.lower(Record.doi) == record_doi.lower())
    ).scalar_one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_record_model(record)


@router.post(
    '/',
    response_model=RecordModel,
)
async def create_record(
        record_in: RecordModelIn,
        metadata_schema: JSONSchema = Depends(get_metadata_schema),
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_WRITE)),
):
    return create_new_record(record_in, metadata_schema, auth)


@router.post(
    '/admin/',
    response_model=RecordModel,
)
async def admin_create_record(
        record_in: RecordModelIn,
        metadata_schema: JSONSchema = Depends(get_metadata_schema),
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_ADMIN)),
):
    return create_new_record(record_in, metadata_schema, auth, True)


@router.put(
    '/{record_id}',
    response_model=RecordModel,
)
async def update_record(
        record_id: str,
        record_in: RecordModelIn,
        metadata_schema: JSONSchema = Depends(get_metadata_schema),
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_WRITE)),
):
    if not (record := Session.get(Record, record_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return set_record(False, record, record_in, metadata_schema, auth)


@router.put(
    '/admin/{record_id}',
    response_model=RecordModel,
)
async def admin_set_record(
        # this route allows a record to be created with an externally
        # generated id, so we must validate that it is a uuid
        record_id: UUID,
        record_in: RecordModelIn,
        metadata_schema: JSONSchema = Depends(get_metadata_schema),
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_ADMIN)),
):
    create = False
    record = Session.get(Record, str(record_id))
    if not record:
        create = True
        record = Record(id=str(record_id))

    return set_record(create, record, record_in, metadata_schema, auth, True)


@router.delete(
    '/{record_id}',
)
async def delete_record(
        record_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_WRITE)),
):
    delete_record_(record_id, auth)


@router.delete(
    '/admin/{record_id}',
)
async def admin_delete_record(
        record_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_ADMIN)),
):
    delete_record_(record_id, auth, True)


@router.post(
    '/{record_id}/tag',
    response_model=TagInstanceModel,
)
async def tag_record(
        record_id: str,
        tag_instance_in: TagInstanceModelIn,
        tag_schema: JSONSchema = Depends(get_tag_schema),
        auth: Authorized = Depends(TagAuthorize()),
):
    if not (record := Session.get(Record, record_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if not (tag := Session.get(Tag, (tag_instance_in.tag_id, TagType.record))):
        raise HTTPException(HTTP_404_NOT_FOUND)

    # only one tag instance per record is allowed
    # update allowed only by the user who did the insert
    if tag.cardinality == TagCardinality.one:
        if record_tag := Session.execute(
                select(RecordTag).
                        where(RecordTag.record_id == record_id).
                        where(RecordTag.tag_id == tag_instance_in.tag_id)
        ).scalar_one_or_none():
            if record_tag.user_id != auth.user_id:
                raise HTTPException(HTTP_409_CONFLICT, 'Cannot update a tag set by another user')
            command = AuditCommand.update
        else:
            command = AuditCommand.insert

    # one tag instance per user per record is allowed
    # update a user's existing tag instance if found
    elif tag.cardinality == TagCardinality.user:
        if record_tag := Session.execute(
                select(RecordTag).
                        where(RecordTag.record_id == record_id).
                        where(RecordTag.tag_id == tag_instance_in.tag_id).
                        where(RecordTag.user_id == auth.user_id)
        ).scalar_one_or_none():
            command = AuditCommand.update
        else:
            command = AuditCommand.insert

    # multiple tag instances are allowed per user per record
    # can only insert/delete
    elif tag.cardinality == TagCardinality.multi:
        command = AuditCommand.insert

    else:
        assert False

    if command == AuditCommand.insert:
        record_tag = RecordTag(
            record_id=record_id,
            tag_id=tag_instance_in.tag_id,
            tag_type=TagType.record,
            user_id=auth.user_id,
            user_name=auth.user_name,
        )

    if record_tag.data != tag_instance_in.data:
        validity = tag_schema.evaluate(JSON(tag_instance_in.data)).output('detailed')
        if not validity['valid']:
            raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, validity)

        if tag.vocabulary_id is not None:
            # it's a keyword tag; we expect the data to conform to the Tag.Keyword schema
            vocab_id = tag_instance_in.data.get('vocabulary')
            keyword_id = tag_instance_in.data.get('keyword')
            if vocab_id != tag.vocabulary_id:
                raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY,
                                    f'Vocabulary {vocab_id} not allowed for tag {tag.id}')
            if not Session.get(VocabularyTerm, (vocab_id, keyword_id)):
                raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY,
                                    f'Vocabulary {vocab_id} does not contain keyword {keyword_id}')

        record_tag.user_name = auth.user_name
        record_tag.data = tag_instance_in.data
        record_tag.timestamp = (timestamp := datetime.now(timezone.utc))
        record_tag.save()

        record.timestamp = timestamp
        record.save()

        touch_parent(record, timestamp)

        create_tag_audit_record(auth, record_tag, timestamp, command)

    return output_tag_instance_model(record_tag)


@router.delete(
    '/{record_id}/tag/{tag_instance_id}',
)
async def untag_record(
        record_id: str,
        tag_instance_id: str,
        auth: Authorized = Depends(UntagAuthorize(TagType.record)),
):
    untag_record_(record_id, tag_instance_id, auth)


@router.delete(
    '/admin/{record_id}/tag/{tag_instance_id}',
)
async def admin_untag_record(
        record_id: str,
        tag_instance_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_ADMIN)),
):
    untag_record_(record_id, tag_instance_id, auth, True)


@router.get(
    '/{record_id}/catalog',
    response_model=Page[CatalogRecordModel],
)
async def list_catalog_records(
        record_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_READ)),
        paginator: Paginator = Depends(partial(Paginator, sort='catalog_id')),
):
    if not (record := Session.get(Record, record_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    stmt = (
        select(CatalogRecord).
        where(CatalogRecord.record_id == record_id)
    )

    return paginator.paginate(
        stmt,
        lambda row: output_catalog_record_model(row.CatalogRecord),
    )


@router.get(
    '/{record_id}/catalog/{catalog_id}',
    response_model=CatalogRecordModel,
)
async def get_catalog_record(
        record_id: str,
        catalog_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_READ)),
):
    if not (catalog_record := Session.get(CatalogRecord, (catalog_id, record_id))):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_catalog_record_model(catalog_record)


@router.get(
    '/{record_id}/audit',
    response_model=Page[AuditModel],
)
async def get_record_audit_log(
        record_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_READ)),
        paginator: Paginator = Depends(partial(Paginator, sort='timestamp')),
):
    audit_subq = union_all(
        select(
            literal_column("'record'").label('table'),
            null().label('tag_id'),
            RecordAudit.id,
            RecordAudit.client_id,
            RecordAudit.user_id,
            RecordAudit.user_name,
            RecordAudit.command,
            RecordAudit.timestamp
        ).where(RecordAudit._id == record_id),
        select(
            literal_column("'record_tag'").label('table'),
            RecordTagAudit._tag_id,
            RecordTagAudit.id,
            RecordTagAudit.client_id,
            RecordTagAudit.user_id,
            RecordTagAudit.user_name,
            RecordTagAudit.command,
            RecordTagAudit.timestamp
        ).where(RecordTagAudit._record_id == record_id)
    ).subquery()

    stmt = select(audit_subq)

    return paginator.paginate(
        stmt,
        lambda row: AuditModel(
            table=row.table,
            tag_id=row.tag_id,
            audit_id=row.id,
            client_id=row.client_id,
            user_id=row.user_id,
            user_name=row.user_name,
            command=row.command,
            timestamp=row.timestamp.isoformat(),
        ),
    )


@router.get(
    '/{record_id}/record_audit/{record_audit_id}',
    response_model=RecordAuditModel,
)
async def get_record_audit_detail(
        record_id: str,
        record_audit_id: int,
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_READ)),
):
    if not (audit := Session.execute(
            select(RecordAudit).
            where(RecordAudit.id == record_audit_id).
            where(RecordAudit._id == record_id)
    ).scalar_one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return RecordAuditModel(
        table='record',
        tag_id=None,
        audit_id=audit.id,
        client_id=audit.client_id,
        user_id=audit.user_id,
        user_name=audit.user_name,
        command=audit.command,
        timestamp=audit.timestamp.isoformat(),
        record_id=audit._id,
        record_doi=audit._doi,
        record_sid=audit._sid,
        record_metadata=audit._metadata,
        record_collection_id=audit._collection_id,
        record_schema_id=audit._schema_id,
        record_parent_id=audit._parent_id,
    )


@router.get(
    '/{record_id}/record_tag_audit/{record_tag_audit_id}',
    response_model=RecordTagAuditModel,
)
async def get_record_tag_audit_detail(
        record_id: str,
        record_tag_audit_id: int,
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_READ)),
):
    if not (audit := Session.execute(
            select(RecordTagAudit).
            where(RecordTagAudit.id == record_tag_audit_id).
            where(RecordTagAudit._record_id == record_id)
    ).scalar_one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return RecordTagAuditModel(
        table='record_tag',
        tag_id=audit._tag_id,
        audit_id=audit.id,
        client_id=audit.client_id,
        user_id=audit.user_id,
        user_name=audit.user_name,
        command=audit.command,
        timestamp=audit.timestamp.isoformat(),
        record_tag_id=audit._id,
        record_tag_record_id=audit._record_id,
        record_tag_user_id=audit._user_id,
        record_tag_user_name=audit._user_name,
        record_tag_data=audit._data,
    )
