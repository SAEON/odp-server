from fastapi import APIRouter, Depends, HTTPException
from jschon import URI
from sqlalchemy import select
from starlette.status import HTTP_404_NOT_FOUND

from odp.api.lib.auth import Authorize
from odp.api.lib.paging import Page, Paginator
from odp.api.models import TagModel
from odp.const import ODPScope
from odp.db import Session
from odp.db.models import Tag
from odp.lib.schema import schema_catalog

router = APIRouter()


@router.get(
    '/',
    response_model=Page[TagModel],
    dependencies=[Depends(Authorize(ODPScope.TAG_READ))],
)
async def list_tags(
        paginator: Paginator = Depends(),
):
    return paginator.paginate(
        select(Tag),
        lambda row: TagModel(
            id=row.Tag.id,
            cardinality=row.Tag.cardinality,
            public=row.Tag.public,
            scope_id=row.Tag.scope_id,
            schema_id=row.Tag.schema_id,
            schema_uri=row.Tag.schema.uri,
            schema_=schema_catalog.get_schema(URI(row.Tag.schema.uri)).value,
        )
    )


@router.get(
    '/{tag_id}',
    response_model=TagModel,
    dependencies=[Depends(Authorize(ODPScope.TAG_READ))],
)
async def get_tag(
        tag_id: str,
):
    tag = Session.execute(
        select(Tag).
        where(Tag.id == tag_id)
    ).scalar_one_or_none()

    if not tag:
        raise HTTPException(HTTP_404_NOT_FOUND)

    return TagModel(
        id=tag.id,
        cardinality=tag.cardinality,
        public=tag.public,
        scope_id=tag.scope_id,
        schema_id=tag.schema_id,
        schema_uri=tag.schema.uri,
        schema_=schema_catalog.get_schema(URI(tag.schema.uri)).value,
    )
