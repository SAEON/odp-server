from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query
from jschon import JSON, URI
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.exc import IntegrityError
from starlette.status import HTTP_404_NOT_FOUND, HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_ENTITY

from odp.api.lib.auth import Authorize, Authorized
from odp.api.lib.paging import Paginator
from odp.api.models import KeywordHierarchyModel, KeywordModel, KeywordModelAdmin, KeywordModelIn, Page
from odp.const import ODPScope
from odp.const.db import AuditCommand, KeywordStatus
from odp.db import Session
from odp.db.models import Keyword, KeywordAudit, Vocabulary
from odp.lib.schema import schema_catalog

router = APIRouter()

_sql = '''
    with recursive ancestors(
        vocabulary_id, parent_id, id, key, data, status, ids, keys
    ) as (
        select vocabulary_id, parent_id, id, key, data, status,
            array[id], array[key]
        from keyword
        where parent_id is null
        union all
        select k.vocabulary_id, k.parent_id, k.id, k.key, k.data, k.status,
            a.ids || k.id, a.keys || k.key
        from keyword k, ancestors a
        where k.parent_id = a.id
    )
    select * from ancestors
'''
""" Hierarchical SQL query that we create below using SQLA select
    and cte constructs, which enable it to be adapted with filters
    and paging. Returns keyword rows supplemented with arrays of
    ancestor ids and keys sorted from root to self, inclusive. """

ancestors = (
    select(
        Keyword,
        array([Keyword.id]).label('ids'),
        array([Keyword.key]).collate('naturalsort').label('keys_'),
    ).where(
        Keyword.parent_id == None
    ).cte(recursive=True)
)
ancestors = ancestors.union_all(
    select(
        Keyword,
        ancestors.c.ids.concat(Keyword.id).label('ids'),
        ancestors.c.keys_.concat(Keyword.key).collate('naturalsort').label('keys_'),
    ).where(
        Keyword.parent_id == ancestors.c.id
    )
)


async def validate_keyword_input(
        vocabulary_id: str,
        keyword_in: KeywordModelIn,
) -> None:
    if not (vocabulary := Session.get(Vocabulary, vocabulary_id)):
        raise HTTPException(
            HTTP_404_NOT_FOUND, 'Vocabulary not found'
        )

    if keyword_in.parent_id is not None:
        if not Session.execute(
                select(Keyword).where(Keyword.vocabulary_id == vocabulary_id).where(Keyword.id == keyword_in.parent_id)
        ).scalar_one_or_none():
            raise HTTPException(
                HTTP_404_NOT_FOUND, 'Parent keyword not found'
            )

    keyword_jsonschema = schema_catalog.get_schema(URI(vocabulary.schema.uri))
    validity = keyword_jsonschema.evaluate(JSON(keyword_in.data)).output('basic')
    if not validity['valid']:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY, validity
        )


class RecurseMode(Enum):
    ALL = 'all'
    APPROVED = 'approved'


def output_keyword_model(keyword: Keyword) -> KeywordModel:
    return KeywordModel(
        vocabulary_id=keyword.vocabulary_id,
        id=keyword.id,
        key=keyword.key,
        data=keyword.data,
        status=keyword.status,
        parent_id=keyword.parent_id,
    )


def output_keyword_hierarchy_model(row) -> KeywordHierarchyModel:
    return KeywordHierarchyModel(
        vocabulary_id=row.vocabulary_id,
        id=row.id,
        key=row.key,
        data=row.data,
        status=row.status,
        parent_id=row.parent_id,
        ids=row.ids,
        keys_=row.keys_,
    )


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
) -> Page[KeywordHierarchyModel]:
    """
    Get a flat list of all keywords, optionally filtered by one or more vocabulary.
    Requires scope `odp.keyword:read_all`.
    """
    stmt = select(ancestors).order_by(ancestors.c.vocabulary_id, ancestors.c.keys_)

    if vocabulary_id:
        stmt = stmt.where(ancestors.c.vocabulary_id.in_(vocabulary_id))

    return paginator.paginate(
        stmt,
        lambda row: output_keyword_hierarchy_model(row),
    )


@router.get(
    '/{keyword_id}',
    dependencies=[Depends(Authorize(ODPScope.KEYWORD_READ_ALL))],
)
async def get_any_keyword(
        keyword_id: int,
) -> KeywordHierarchyModel:
    """
    Get any keyword by id. Requires scope `odp.keyword:read_all`.
    """
    if not (row := Session.execute(
            select(ancestors).where(ancestors.c.id == keyword_id)
    ).one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_keyword_hierarchy_model(row)


@router.get(
    '/{vocabulary_id}/',
    dependencies=[Depends(Authorize(ODPScope.KEYWORD_READ))],
)
async def list_keywords(
        vocabulary_id: str,
        parent_key: str = None,
        include_proposed: bool = False,
        paginator: Paginator = Depends(),
) -> Page[KeywordHierarchyModel]:
    """
    Get a flat list of approved and (optionally) proposed keywords for a vocabulary.
    Specify `parent_key` to return only immediate child keywords of a keyword.
    Requires scope `odp.keyword:read`.
    """
    if not Session.get(Vocabulary, vocabulary_id):
        raise HTTPException(
            HTTP_404_NOT_FOUND, 'Vocabulary not found'
        )

    include_statuses = [KeywordStatus.approved]
    if include_proposed:
        include_statuses += [KeywordStatus.proposed]

    # Note: If a parent keyword is not approved but has approved children,
    # we should ideally not include those children in the response. This is,
    # however, an unlikely scenario, and we are simply returning all approved
    # keywords from anywhere in the hierarchy.

    stmt = (
        select(ancestors).
        where(ancestors.c.vocabulary_id == vocabulary_id).
        where(ancestors.c.status.in_(include_statuses))
    )

    if parent_key is not None:
        if not (parent_id := Session.execute(
                select(Keyword.id).where(Keyword.vocabulary_id == vocabulary_id).where(Keyword.key == parent_key)
        ).scalar_one_or_none()):
            raise HTTPException(
                HTTP_404_NOT_FOUND, 'Parent keyword not found'
            )
        stmt = stmt.where(ancestors.c.parent_id == parent_id)

    return paginator.paginate(
        stmt,
        lambda row: output_keyword_hierarchy_model(row),
    )


@router.get(
    '/{vocabulary_id}/{key}',
    dependencies=[Depends(Authorize(ODPScope.KEYWORD_READ))],
)
async def get_keyword(
        vocabulary_id: str,
        key: str,
) -> KeywordHierarchyModel:
    """
    Get an approved keyword. Requires scope `odp.keyword:read`.
    """
    stmt = (
        select(ancestors).
        where(ancestors.c.vocabulary_id == vocabulary_id).
        where(ancestors.c.key == key)
    )
    found = False
    if row := Session.execute(stmt).one_or_none():
        found = row.status == KeywordStatus.approved

    if not found:
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_keyword_hierarchy_model(row)


@router.post(
    '/{vocabulary_id}/',
)
async def suggest_keyword(
        vocabulary_id: str,
        keyword_in: KeywordModelIn,
        auth: Authorized = Depends(Authorize(ODPScope.KEYWORD_SUGGEST)),
        _=Depends(validate_keyword_input),
) -> KeywordModel:
    """
    Create a keyword with status `proposed`. Requires scope `odp.keyword:suggest`.
    """
    return _create_keyword(
        vocabulary_id,
        keyword_in.key,
        keyword_in.data,
        KeywordStatus.proposed,
        keyword_in.parent_id,
        auth,
    )


@router.put(
    '/{vocabulary_id}/',
)
async def create_keyword(
        vocabulary_id: str,
        keyword_in: KeywordModelAdmin,
        auth: Authorized = Depends(Authorize(ODPScope.KEYWORD_ADMIN)),
        _=Depends(validate_keyword_input),
) -> KeywordModel:
    """
    Create a keyword. Requires scope `odp.keyword:admin`.
    """
    return _create_keyword(
        vocabulary_id,
        keyword_in.key,
        keyword_in.data,
        keyword_in.status,
        keyword_in.parent_id,
        auth,
    )


def _create_keyword(
        vocabulary_id: str,
        key: str,
        data: dict,
        status: KeywordStatus,
        parent_id: int | None,
        auth: Authorized,
) -> KeywordModel:
    if Session.execute(
            select(Keyword).where(Keyword.vocabulary_id == vocabulary_id).where(Keyword.key == key)
    ).scalar_one_or_none():
        raise HTTPException(
            HTTP_409_CONFLICT, f"Keyword '{key}' already exists"
        )

    keyword = Keyword(
        vocabulary_id=vocabulary_id,
        key=key,
        data=data,
        status=status,
        parent_id=parent_id,
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
    '/{vocabulary_id}/{keyword_id}',
)
async def update_keyword(
        vocabulary_id: str,
        keyword_id: int,
        keyword_in: KeywordModelAdmin,
        auth: Authorized = Depends(Authorize(ODPScope.KEYWORD_ADMIN)),
        _=Depends(validate_keyword_input),
) -> KeywordModel | None:
    """
    Update a keyword. Requires scope `odp.keyword:admin`.
    """
    if not (keyword := Session.get(Keyword, (vocabulary_id, keyword_id))):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if (
            keyword.key != keyword_in.key or
            keyword.data != keyword_in.data or
            keyword.status != keyword_in.status or
            keyword.parent_id != keyword_in.parent_id
    ):
        keyword.key = keyword_in.key
        keyword.data = keyword_in.data
        keyword.status = keyword_in.status
        keyword.parent_id = keyword_in.parent_id

        keyword.save()

        create_audit_record(
            auth,
            keyword,
            datetime.now(timezone.utc),
            AuditCommand.update,
        )

        return output_keyword_model(keyword)


@router.delete(
    '/{vocabulary_id}/{keyword_id}',
)
async def delete_keyword(
        vocabulary_id: str,
        keyword_id: int,
        auth: Authorized = Depends(Authorize(ODPScope.KEYWORD_ADMIN)),
) -> None:
    """
    Delete a keyword. Requires scope `odp.keyword:admin`.
    """
    if not (keyword := Session.get(Keyword, (vocabulary_id, keyword_id))):
        raise HTTPException(HTTP_404_NOT_FOUND)

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
            HTTP_422_UNPROCESSABLE_ENTITY, f"Keyword '{keyword_id}' has child keywords"
        ) from e
