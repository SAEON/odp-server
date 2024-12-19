from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from jschon import JSON, URI
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from starlette.status import HTTP_404_NOT_FOUND, HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_ENTITY

from odp.api.lib.auth import Authorize, Authorized
from odp.api.lib.paging import Paginator
from odp.api.models import KeywordHierarchyModel, KeywordModel, KeywordModelAdmin, KeywordModelIn, Page
from odp.const import ODPScope
from odp.const.db import AuditCommand, KeywordStatus, SchemaType
from odp.db import Session
from odp.db.models import Keyword, KeywordAudit, Schema
from odp.lib.schema import schema_catalog

router = APIRouter()


def get_child_schema(parent: Keyword) -> Schema | None:
    """Get the validating schema for keywords in the given parent vocabulary."""
    if parent is None:
        return None

    if parent.child_schema_id:
        return parent.child_schema

    return get_child_schema(parent.parent)


async def validate_keyword_input(
        keyword_id: str,
        keyword_in: KeywordModelIn,
) -> None:
    if not keyword_id.startswith(keyword_in.parent_id) or keyword_id == keyword_in.parent_id:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY, f"'{keyword_in.parent_id}' cannot be a parent of '{keyword_id}'"
        )

    if not (parent := Session.get(Keyword, keyword_in.parent_id)):
        raise HTTPException(
            HTTP_404_NOT_FOUND, f"Parent keyword '{keyword_in.parent_id}' does not exist"
        )

    keyword_jsonschema = schema_catalog.get_schema(URI(
        get_child_schema(parent).uri
    ))

    validity = keyword_jsonschema.evaluate(JSON(keyword_in.data)).output('basic')
    if not validity['valid']:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY, validity
        )


class RecurseMode(Enum):
    ALL = 'all'
    APPROVED = 'approved'


def output_keyword_model(
        keyword: Keyword,
        *,
        recurse: RecurseMode = None,
) -> KeywordModel | KeywordHierarchyModel:
    cls = KeywordHierarchyModel if recurse else KeywordModel

    kwargs = dict(
        vocabulary_id=keyword.vocabulary_id,
        id=keyword.id,
        key=keyword.key,
        data=keyword.data,
        status=keyword.status,
        parent_id=keyword.parent_id,
        parent_key=keyword.parent.key if keyword.parent_id else None,
        schema_id=keyword.vocabulary.schema_id,
    )

    if recurse:
        kwargs |= dict(
            child_keywords=[
                output_keyword_model(child, recurse=recurse)
                for child in keyword.children
                if recurse == RecurseMode.ALL or child.status == KeywordStatus.approved
            ]
        )

    return cls(**kwargs)


def create_audit_record(
        auth: Authorized,
        keyword: Keyword,
        timestamp: datetime,
        command: AuditCommand,
) -> None:
    KeywordAudit(
        client_id=auth.client_id,
        user_id=auth.user_id,
        command=command,
        timestamp=timestamp,
        _vocabulary_id=keyword.vocabulary_id,
        _id=keyword.id,
        _key=keyword.key,
        _data=keyword.data,
        _status=keyword.status,
        _parent_id=keyword.parent_id,
    ).save()


@router.get(
    '/',
    dependencies=[Depends(Authorize(ODPScope.KEYWORD_READ_ALL))],
)
async def list_all_keywords(
        vocabulary_id: list[str] = Query(None, title='Filter by vocabulary(-ies)'),
        paginator: Paginator = Depends(),
) -> Page[KeywordModel]:
    """
    Get a flat list of all keywords, optionally filtered by one or more vocabulary.
    Requires scope `odp.keyword:read_all`.
    """
    stmt = select(Keyword)

    if vocabulary_id:
        stmt = stmt.where(Keyword.vocabulary_id.in_(vocabulary_id))

    return paginator.paginate(
        stmt,
        lambda row: output_keyword_model(row.Keyword),
    )


@router.get(
    '/{keyword_id}',
    dependencies=[Depends(Authorize(ODPScope.KEYWORD_READ_ALL))],
)
async def get_any_keyword(
        keyword_id: int,
        recurse: bool = Query(False, title='Populate child keywords, recursively'),
) -> KeywordHierarchyModel | KeywordModel:
    """
    Get any keyword by id. Requires scope `odp.keyword:read_all`.
    """
    if not (keyword := Session.execute(
            select(Keyword).where(Keyword.id == keyword_id)
    ).scalar_one_or_none()):
        raise HTTPException(
            HTTP_404_NOT_FOUND, f'Keyword id {keyword_id} not found'
        )

    return output_keyword_model(keyword, recurse=RecurseMode.ALL if recurse else None)


@router.get(
    '/{vocabulary_id}/',
    dependencies=[Depends(Authorize(ODPScope.KEYWORD_READ))],
)
async def list_keywords(
        vocabulary_id: str,
        paginator: Paginator = Depends(),
) -> Page[KeywordModel]:
    """
    Get a flat list of approved keywords for a vocabulary. Requires scope `odp.keyword:read`.
    """
    # Note: If a parent keyword is not approved but has approved children,
    # we should ideally not include those children in the response. We are,
    # however, simply returning all approved keywords from anywhere in the
    # hierarchy. In this (edge) case, the caller will see such child keywords
    # as being orphaned.
    stmt = (
        select(Keyword).
        where(Keyword.vocabulary_id == vocabulary_id).
        where(Keyword.status == KeywordStatus.approved)
    )

    return paginator.paginate(
        stmt,
        lambda row: output_keyword_model(row.Keyword),
    )


@router.get(
    '/{vocabulary_id}/{key}',
    dependencies=[Depends(Authorize(ODPScope.KEYWORD_READ))],
)
async def get_keyword(
        vocabulary_id: str,
        key: str,
        recurse: bool = Query(False, title='Populate child keywords, recursively'),
) -> KeywordHierarchyModel | KeywordModel:
    """
    Get an approved keyword, optionally with child keywords. Requires scope `odp.keyword:read`.
    """
    found = False
    if keyword := Session.execute(
            select(Keyword).where(Keyword.vocabulary_id == vocabulary_id).where(Keyword.key == key)
    ).scalar_one_or_none():
        found = keyword.status == KeywordStatus.approved

    if not found:
        raise HTTPException(
            HTTP_404_NOT_FOUND, f"Keyword '{key}' not found"
        )

    return output_keyword_model(keyword, recurse=RecurseMode.APPROVED if recurse else None)


@router.post(
    '/{keyword_id:path}',
)
async def suggest_keyword(
        keyword_in: KeywordModelIn,
        keyword_id: str = Path(..., title='Keyword identifier'),
        auth: Authorized = Depends(Authorize(ODPScope.KEYWORD_SUGGEST)),
        _=Depends(validate_keyword_input),
) -> KeywordModel:
    """
    Create a keyword with status `proposed`. Requires scope `odp.keyword:suggest`.
    """
    if Session.get(Keyword, keyword_id):
        raise HTTPException(
            HTTP_409_CONFLICT, f"Keyword '{keyword_id}' already exists"
        )

    keyword = Keyword(
        id=keyword_id,
        data=keyword_in.data,
        status=KeywordStatus.proposed,
        parent_id=keyword_in.parent_id,
    )
    keyword.save()

    create_audit_record(
        auth,
        keyword,
        datetime.now(timezone.utc),
        AuditCommand.insert,
    )

    return output_keyword_model(keyword)


@router.put(
    '/{keyword_id:path}',
)
async def set_keyword(
        keyword_in: KeywordModelAdmin,
        keyword_id: str = Path(..., title='Keyword identifier'),
        auth: Authorized = Depends(Authorize(ODPScope.KEYWORD_ADMIN)),
        _=Depends(validate_keyword_input),
) -> KeywordModel:
    """
    Create or update a keyword. Requires scope `odp.keyword:admin`.
    """
    if keyword := Session.get(Keyword, keyword_id):
        command = AuditCommand.update
    else:
        command = AuditCommand.insert
        keyword = Keyword(
            id=keyword_id,
            parent_id=keyword_in.parent_id,
        )

    if (
            keyword.data != keyword_in.data or
            keyword.status != keyword_in.status or
            keyword.child_schema_id != keyword_in.child_schema_id
    ):
        keyword.data = keyword_in.data
        keyword.status = keyword_in.status
        keyword.child_schema_id = keyword_in.child_schema_id
        keyword.child_schema_type = SchemaType.keyword if keyword_in.child_schema_id else None

        keyword.save()

        create_audit_record(
            auth,
            keyword,
            datetime.now(timezone.utc),
            command,
        )

    return output_keyword_model(keyword)


@router.delete(
    '/{keyword_id:path}',
)
async def delete_keyword(
        keyword_id: str = Path(..., title='Keyword identifier'),
        auth: Authorized = Depends(Authorize(ODPScope.KEYWORD_ADMIN)),
) -> None:
    """
    Delete a keyword. Requires scope `odp.keyword:admin`.
    """
    if not (keyword := Session.get(Keyword, keyword_id)):
        raise HTTPException(
            HTTP_404_NOT_FOUND, f"Keyword '{keyword_id}' does not exist"
        )

    create_audit_record(
        auth,
        keyword,
        datetime.now(timezone.utc),
        AuditCommand.delete,
    )

    try:
        keyword.delete()

    except IntegrityError as e:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY, f"Keyword '{keyword_id}' cannot be deleted as it has sub-keywords"
        ) from e
