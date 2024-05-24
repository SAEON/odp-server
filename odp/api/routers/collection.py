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
from odp.api.lib.paging import Paginator
from odp.api.lib.schema import get_tag_schema
from odp.api.lib.utils import output_tag_instance_model
from odp.api.models import (
    AuditModel,
    CollectionAuditModel,
    CollectionModel,
    CollectionModelIn,
    CollectionTagAuditModel,
    Page,
    TagInstanceModel,
    TagInstanceModelIn,
)
from odp.const import DOI_PREFIX, ODPScope
from odp.const.db import AuditCommand, TagCardinality, TagType
from odp.db import Session
from odp.db.models import Collection, CollectionAudit, CollectionTag, CollectionTagAudit, Record, Tag, User

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
        role_ids=[role.id for role in result.Collection.roles],
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
        command=command,
        timestamp=timestamp,
        _id=collection_tag.id,
        _collection_id=collection_tag.collection_id,
        _tag_id=collection_tag.tag_id,
        _user_id=collection_tag.user_id,
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
    if auth.object_ids != '*':
        stmt = stmt.where(Collection.id.in_(auth.object_ids))

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
    auth.enforce_constraint([collection_id])

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
    auth.enforce_constraint('*')

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
        select(Collection, literal_column('0').label('count')).
        where(Collection.id == collection.id)
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
    auth.enforce_constraint([collection_id])

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
    auth.enforce_constraint([collection_id])

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
    auth.enforce_constraint([collection_id])

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
        )

    if collection_tag.data != tag_instance_in.data:
        validity = tag_schema.evaluate(JSON(tag_instance_in.data)).output('detailed')
        if not validity['valid']:
            raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, validity)

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
    auth.enforce_constraint([collection_id])

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
    auth.enforce_constraint([collection_id])

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
    auth.enforce_constraint([collection_id])

    audit_subq = union_all(
        select(
            literal_column("'collection'").label('table'),
            null().label('tag_id'),
            CollectionAudit.id,
            CollectionAudit.client_id,
            CollectionAudit.user_id,
            CollectionAudit.command,
            CollectionAudit.timestamp
        ).where(CollectionAudit._id == collection_id),
        select(
            literal_column("'collection_tag'").label('table'),
            CollectionTagAudit._tag_id,
            CollectionTagAudit.id,
            CollectionTagAudit.client_id,
            CollectionTagAudit.user_id,
            CollectionTagAudit.command,
            CollectionTagAudit.timestamp
        ).where(CollectionTagAudit._collection_id == collection_id)
    ).subquery()

    stmt = (
        select(audit_subq, User.name.label('user_name')).
        outerjoin(User, audit_subq.c.user_id == User.id)
    )

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
    auth.enforce_constraint([collection_id])

    if not (row := Session.execute(
        select(CollectionAudit, User.name.label('user_name')).
        outerjoin(User, CollectionAudit.user_id == User.id).
        where(CollectionAudit.id == collection_audit_id).
        where(CollectionAudit._id == collection_id)
    ).one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return CollectionAuditModel(
        table='collection',
        tag_id=None,
        audit_id=row.CollectionAudit.id,
        client_id=row.CollectionAudit.client_id,
        user_id=row.CollectionAudit.user_id,
        user_name=row.user_name,
        command=row.CollectionAudit.command,
        timestamp=row.CollectionAudit.timestamp.isoformat(),
        collection_id=row.CollectionAudit._id,
        collection_key=row.CollectionAudit._key,
        collection_name=row.CollectionAudit._name,
        collection_doi_key=row.CollectionAudit._doi_key,
        collection_provider_id=row.CollectionAudit._provider_id,
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
    auth.enforce_constraint([collection_id])

    audit_user_alias = aliased(User)
    tag_user_alias = aliased(User)

    if not (row := Session.execute(
        select(
            CollectionTagAudit,
            audit_user_alias.name.label('audit_user_name'),
            tag_user_alias.name.label('tag_user_name')
        ).
        outerjoin(audit_user_alias, CollectionTagAudit.user_id == audit_user_alias.id).
        outerjoin(tag_user_alias, CollectionTagAudit._user_id == tag_user_alias.id).
        where(CollectionTagAudit.id == collection_tag_audit_id).
        where(CollectionTagAudit._collection_id == collection_id)
    ).one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return CollectionTagAuditModel(
        table='collection_tag',
        tag_id=row.CollectionTagAudit._tag_id,
        audit_id=row.CollectionTagAudit.id,
        client_id=row.CollectionTagAudit.client_id,
        user_id=row.CollectionTagAudit.user_id,
        user_name=row.audit_user_name,
        command=row.CollectionTagAudit.command,
        timestamp=row.CollectionTagAudit.timestamp.isoformat(),
        collection_tag_id=row.CollectionTagAudit._id,
        collection_tag_collection_id=row.CollectionTagAudit._collection_id,
        collection_tag_user_id=row.CollectionTagAudit._user_id,
        collection_tag_user_name=row.tag_user_name,
        collection_tag_data=row.CollectionTagAudit._data,
    )
