from datetime import datetime, timezone
from functools import partial
from random import randint

from fastapi import APIRouter, Depends, HTTPException
from jschon import JSON, JSONSchema
from sqlalchemy import func, literal_column, null, select, union_all
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import aliased
from starlette.status import HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND, HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_ENTITY

from odp.api.lib.auth import Authorize, Authorized, TagAuthorize, UntagAuthorize
from odp.api.lib.paging import Page, Paginator
from odp.api.lib.schema import get_tag_schema
from odp.api.lib.utils import output_tag_instance_model
from odp.api.models import (AuditModel, CollectionAuditModel, CollectionModel, CollectionModelIn, CollectionTagAuditModel, TagInstanceModel,
                            TagInstanceModelIn)
from odp.const import DOI_PREFIX, ODPScope
from odp.const.db import AuditCommand, TagCardinality, TagType
from odp.db import Session
from odp.db.models import Collection, CollectionAudit, CollectionTag, CollectionTagAudit, Record, Tag

router = APIRouter()


def output_collection_model(result) -> CollectionModel:
    return CollectionModel(
        id=result.Collection.id,
        key=result.Collection.key,
        name=result.Collection.name,
        doi_key=result.Collection.doi_key,
        provider_id=result.Collection.provider_id,
        provider_key=result.Collection.provider.key,
        record_count=result.count,
        tags=[
            output_tag_instance_model(collection_tag)
            for collection_tag in result.Collection.tags
        ],
        timestamp=result.Collection.timestamp.isoformat(),
    )


def create_audit_record(
        auth: Authorized,
        collection: Collection,
        timestamp: datetime,
        command: AuditCommand,
) -> None:
    CollectionAudit(
        client_id=auth.client_id,
        user_id=auth.user_id,
        user_name=auth.user_name,
        command=command,
        timestamp=timestamp,
        _id=collection.id,
        _key=collection.key,
        _name=collection.name,
        _doi_key=collection.doi_key,
        _provider_id=collection.provider_id,
    ).save()


def create_tag_audit_record(
        auth: Authorized,
        collection_tag: CollectionTag,
        timestamp: datetime,
        command: AuditCommand,
) -> None:
    CollectionTagAudit(
        client_id=auth.client_id,
        user_id=auth.user_id,
        user_name=auth.user_name,
        command=command,
        timestamp=timestamp,
        _id=collection_tag.id,
        _collection_id=collection_tag.collection_id,
        _tag_id=collection_tag.tag_id,
        _user_id=collection_tag.user_id,
        _user_name=collection_tag.user_name,
        _data=collection_tag.data,
    ).save()


@router.get(
    '/',
    response_model=Page[CollectionModel],
)
async def list_collections(
        auth: Authorized = Depends(Authorize(ODPScope.COLLECTION_READ)),
        paginator: Paginator = Depends(partial(Paginator, sort='key')),
):
    stmt = (
        select(Collection, func.count(Record.id)).
        outerjoin(Record).
        group_by(Collection)
    )

    return paginator.paginate(
        stmt,
        lambda row: output_collection_model(row),
        sort_model=Collection,
    )


@router.get(
    '/{collection_id}',
    response_model=CollectionModel,
)
async def get_collection(
        collection_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.COLLECTION_READ)),
):
    stmt = (
        select(Collection, func.count(Record.id)).
        outerjoin(Record).
        where(Collection.id == collection_id).
        group_by(Collection)
    )

    if not (result := Session.execute(stmt).one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_collection_model(result)


@router.post(
    '/',
    response_model=CollectionModel,
)
async def create_collection(
        collection_in: CollectionModelIn,
        auth: Authorized = Depends(Authorize(ODPScope.COLLECTION_ADMIN)),
):
    if Session.execute(
            select(Collection).
            where(Collection.key == collection_in.key)
    ).first() is not None:
        raise HTTPException(HTTP_409_CONFLICT, 'Collection key is already in use')

    collection = Collection(
        key=collection_in.key,
        name=collection_in.name,
        doi_key=collection_in.doi_key,
        provider_id=collection_in.provider_id,
        timestamp=(timestamp := datetime.now(timezone.utc)),
    )
    collection.save()
    create_audit_record(auth, collection, timestamp, AuditCommand.insert)

    result = Session.execute(
        select(Collection, func.count(Record.id)).
        outerjoin(Record).
        where(Collection.id == collection.id).
        group_by(Collection)
    ).first()

    return output_collection_model(result)


@router.put(
    '/{collection_id}',
)
async def update_collection(
        collection_id: str,
        collection_in: CollectionModelIn,
        auth: Authorized = Depends(Authorize(ODPScope.COLLECTION_ADMIN)),
):
    if not (collection := Session.get(Collection, collection_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if Session.execute(
            select(Collection).
            where(Collection.id != collection_id).
            where(Collection.key == collection_in.key)
    ).first() is not None:
        raise HTTPException(HTTP_409_CONFLICT, 'Collection key is already in use')

    if (
            collection.key != collection_in.key or
            collection.name != collection_in.name or
            collection.doi_key != collection_in.doi_key or
            collection.provider_id != collection_in.provider_id
    ):
        collection.key = collection_in.key
        collection.name = collection_in.name
        collection.doi_key = collection_in.doi_key
        collection.provider_id = collection_in.provider_id
        collection.timestamp = (timestamp := datetime.now(timezone.utc))
        collection.save()
        create_audit_record(auth, collection, timestamp, AuditCommand.update)


@router.delete(
    '/{collection_id}',
)
async def delete_collection(
        collection_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.COLLECTION_ADMIN)),
):
    if not (collection := Session.get(Collection, collection_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    try:
        collection.delete()
    except IntegrityError as e:
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'A non-empty collection cannot be deleted.') from e

    create_audit_record(auth, collection, datetime.now(timezone.utc), AuditCommand.delete)


@router.post(
    '/{collection_id}/tag',
    response_model=TagInstanceModel,
)
async def tag_collection(
        collection_id: str,
        tag_instance_in: TagInstanceModelIn,
        tag_schema: JSONSchema = Depends(get_tag_schema),
        auth: Authorized = Depends(TagAuthorize()),
):
    if not (collection := Session.get(Collection, collection_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if not (tag := Session.get(Tag, (tag_instance_in.tag_id, TagType.collection))):
        raise HTTPException(HTTP_404_NOT_FOUND)

    # only one tag instance per collection is allowed
    # update allowed only by the user who did the insert
    if tag.cardinality == TagCardinality.one:
        if collection_tag := Session.execute(
                select(CollectionTag).
                where(CollectionTag.collection_id == collection_id).
                where(CollectionTag.tag_id == tag_instance_in.tag_id)
        ).scalar_one_or_none():
            if collection_tag.user_id != auth.user_id:
                raise HTTPException(HTTP_409_CONFLICT, 'Cannot update a tag set by another user')
            command = AuditCommand.update
        else:
            command = AuditCommand.insert

    # one tag instance per user per collection is allowed
    # update a user's existing tag instance if found
    elif tag.cardinality == TagCardinality.user:
        if collection_tag := Session.execute(
                select(CollectionTag).
                where(CollectionTag.collection_id == collection_id).
                where(CollectionTag.tag_id == tag_instance_in.tag_id).
                where(CollectionTag.user_id == auth.user_id)
        ).scalar_one_or_none():
            command = AuditCommand.update
        else:
            command = AuditCommand.insert

    # multiple tag instances are allowed per user per collection
    # can only insert/delete
    elif tag.cardinality == TagCardinality.multi:
        command = AuditCommand.insert

    else:
        assert False

    if command == AuditCommand.insert:
        collection_tag = CollectionTag(
            collection_id=collection_id,
            tag_id=tag_instance_in.tag_id,
            tag_type=TagType.collection,
            user_id=auth.user_id,
            user_name=auth.user_name,
        )

    if collection_tag.data != tag_instance_in.data:
        validity = tag_schema.evaluate(JSON(tag_instance_in.data)).output('detailed')
        if not validity['valid']:
            raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, validity)

        collection_tag.user_name = auth.user_name
        collection_tag.data = tag_instance_in.data
        collection_tag.timestamp = (timestamp := datetime.now(timezone.utc))
        collection_tag.save()

        collection.timestamp = timestamp
        collection.save()

        create_tag_audit_record(auth, collection_tag, timestamp, command)

    return output_tag_instance_model(collection_tag)


@router.delete(
    '/{collection_id}/tag/{tag_instance_id}',
)
async def untag_collection(
        collection_id: str,
        tag_instance_id: str,
        auth: Authorized = Depends(UntagAuthorize(TagType.collection)),
):
    _untag_collection(collection_id, tag_instance_id, auth)


@router.delete(
    '/admin/{collection_id}/tag/{tag_instance_id}',
)
async def admin_untag_collection(
        collection_id: str,
        tag_instance_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.COLLECTION_ADMIN)),
):
    _untag_collection(collection_id, tag_instance_id, auth, True)


def _untag_collection(
        collection_id: str,
        tag_instance_id: str,
        auth: Authorized,
        ignore_user_id: bool = False,
) -> None:
    if not (collection := Session.get(Collection, collection_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if not (collection_tag := Session.execute(
        select(CollectionTag).
        where(CollectionTag.id == tag_instance_id).
        where(CollectionTag.collection_id == collection_id)
    ).scalar_one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if not ignore_user_id and collection_tag.user_id != auth.user_id:
        raise HTTPException(HTTP_403_FORBIDDEN)

    collection_tag.delete()

    collection.timestamp = (timestamp := datetime.now(timezone.utc))
    collection.save()

    create_tag_audit_record(auth, collection_tag, timestamp, AuditCommand.delete)


@router.get(
    '/{collection_id}/doi/new',
    response_model=str,
)
async def get_new_doi(
        collection_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.COLLECTION_READ)),
):
    if not (collection := Session.get(Collection, collection_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if not (doi_key := collection.doi_key):
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'The collection does not have a DOI key')

    while True:
        num = randint(0, 99999999)
        doi = f'{DOI_PREFIX}/{doi_key}.{num:08}'
        if Session.execute(select(Record).where(func.lower(Record.doi) == doi.lower())).first() is None:
            break

    return doi


@router.get(
    '/{collection_id}/audit',
    response_model=Page[AuditModel],
)
async def get_collection_audit_log(
        collection_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.COLLECTION_READ)),
        paginator: Paginator = Depends(partial(Paginator, sort='timestamp')),
):
    audit_subq = union_all(
        select(
            literal_column("'collection'").label('table'),
            null().label('tag_id'),
            CollectionAudit.id,
            CollectionAudit.client_id,
            CollectionAudit.user_id,
            CollectionAudit.user_name,
            CollectionAudit.command,
            CollectionAudit.timestamp
        ).where(CollectionAudit._id == collection_id),
        select(
            literal_column("'collection_tag'").label('table'),
            CollectionTagAudit._tag_id,
            CollectionTagAudit.id,
            CollectionTagAudit.client_id,
            CollectionTagAudit.user_id,
            CollectionTagAudit.user_name,
            CollectionTagAudit.command,
            CollectionTagAudit.timestamp
        ).where(CollectionTagAudit._collection_id == collection_id)
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
    '/{collection_id}/collection_audit/{collection_audit_id}',
    response_model=CollectionAuditModel,
)
async def get_collection_audit_detail(
        collection_id: str,
        collection_audit_id: int,
        auth: Authorized = Depends(Authorize(ODPScope.COLLECTION_READ)),
):
    if not (audit := Session.execute(
        select(CollectionAudit).
        where(CollectionAudit.id == collection_audit_id).
        where(CollectionAudit._id == collection_id)
    ).scalar_one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return CollectionAuditModel(
        table='collection',
        tag_id=None,
        audit_id=audit.id,
        client_id=audit.client_id,
        user_id=audit.user_id,
        user_name=audit.user_name,
        command=audit.command,
        timestamp=audit.timestamp.isoformat(),
        collection_id=audit._id,
        collection_key=audit._key,
        collection_name=audit._name,
        collection_doi_key=audit._doi_key,
        collection_provider_id=audit._provider_id,
    )


@router.get(
    '/{collection_id}/collection_tag_audit/{collection_tag_audit_id}',
    response_model=CollectionTagAuditModel,
)
async def get_collection_tag_audit_detail(
        collection_id: str,
        collection_tag_audit_id: int,
        auth: Authorized = Depends(Authorize(ODPScope.COLLECTION_READ)),
):
    if not (audit := Session.execute(
        select(CollectionTagAudit).
        where(CollectionTagAudit.id == collection_tag_audit_id).
        where(CollectionTagAudit._collection_id == collection_id)
    ).scalar_one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return CollectionTagAuditModel(
        table='collection_tag',
        tag_id=audit._tag_id,
        audit_id=audit.id,
        client_id=audit.client_id,
        user_id=audit.user_id,
        user_name=audit.user_name,
        command=audit.command,
        timestamp=audit.timestamp.isoformat(),
        collection_tag_id=audit._id,
        collection_tag_collection_id=audit._collection_id,
        collection_tag_user_id=audit._user_id,
        collection_tag_user_name=audit._user_name,
        collection_tag_data=audit._data,
    )
