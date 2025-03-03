import re
from datetime import date
from enum import Enum
from functools import partial
from math import ceil
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import RedirectResponse
from jschon import JSONPointer
from jschon.exc import JSONPointerMalformedError, JSONPointerReferenceError
from pydantic import Json
from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.orm import aliased, load_only
from starlette.status import HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_ENTITY

from odp.api.lib.auth import Authorize
from odp.api.lib.datacite import get_datacite_client
from odp.api.lib.paging import Paginator
from odp.api.lib.utils import output_published_record_model
from odp.api.models import (
    CatalogModel,
    CatalogModelWithData,
    Page,
    PublishedDataCiteRecordModel,
    PublishedSAEONRecordModel,
    RetractedRecordModel,
    SearchResult,
)
from odp.const import DOI_REGEX, ODPCatalog, ODPScope
from odp.db import Session
from odp.db.models import Catalog, CatalogRecord, CatalogRecordFacet, PublishedRecord, Record
from odp.lib.datacite import DataciteClient, DataciteError

router = APIRouter()


class SearchResultSort(str, Enum):
    TIMESTAMP_DESC = 'timestamp desc'
    RANK_DESC = 'rank desc'


@router.get(
    '/',
    response_model=Page[CatalogModel],
    dependencies=[Depends(Authorize(ODPScope.CATALOG_READ))],
)
async def list_catalogs(
        paginator: Paginator = Depends(),
):
    stmt = (
        select(Catalog, func.count(CatalogRecord.catalog_id)).
        options(load_only(Catalog.id, Catalog.url)).
        outerjoin(CatalogRecord, and_(Catalog.id == CatalogRecord.catalog_id, CatalogRecord.published)).
        group_by(Catalog)
    )

    return paginator.paginate(
        stmt,
        lambda row: CatalogModel(
            id=row.Catalog.id,
            url=row.Catalog.url,
            record_count=row.count,
        )
    )


@router.get(
    '/{catalog_id}',
    response_model=CatalogModelWithData,
    dependencies=[Depends(Authorize(ODPScope.CATALOG_READ))],
)
async def get_catalog(
        catalog_id: str,
):
    stmt = (
        select(Catalog, func.count(CatalogRecord.catalog_id)).
        outerjoin(CatalogRecord, and_(Catalog.id == CatalogRecord.catalog_id, CatalogRecord.published)).
        group_by(Catalog).
        where(Catalog.id == catalog_id)
    )

    if not (result := Session.execute(stmt).one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return CatalogModelWithData(
        id=result.Catalog.id,
        url=result.Catalog.url,
        data=result.Catalog.data,
        timestamp=result.Catalog.timestamp.isoformat() if result.Catalog.timestamp else None,
        record_count=result.count,
    )


@router.get(
    '/{catalog_id}/records',
    response_model=Page[PublishedSAEONRecordModel | PublishedDataCiteRecordModel | RetractedRecordModel],
    dependencies=[Depends(Authorize(ODPScope.CATALOG_READ))],
)
async def list_records(
        catalog_id: str,
        include_nonsearchable: bool = False,
        include_retracted: bool = False,
        updated_since: date = None,
        paginator: Paginator = Depends(partial(Paginator, sort='timestamp')),
):
    if not Session.get(Catalog, catalog_id):
        raise HTTPException(HTTP_404_NOT_FOUND)

    stmt = (
        select(CatalogRecord)
        .where(CatalogRecord.catalog_id == catalog_id)
    )

    if not include_nonsearchable:
        stmt = stmt.where(or_(CatalogRecord.searchable == None, CatalogRecord.searchable))

    if include_retracted:
        stmt = stmt.join(PublishedRecord, CatalogRecord.record_id == PublishedRecord.id)
    else:
        stmt = stmt.where(CatalogRecord.published)

    if updated_since:
        stmt = stmt.where(CatalogRecord.timestamp >= updated_since)

    return paginator.paginate(
        stmt,
        lambda row: output_published_record_model(row.CatalogRecord) if row.CatalogRecord.published
        else RetractedRecordModel(id=row.CatalogRecord.record_id),
    )


@router.get(
    '/{catalog_id}/search',
    response_model=SearchResult,
    dependencies=[Depends(Authorize(ODPScope.CATALOG_SEARCH))],
    description="Search a catalog's published records.",
)
async def search_records(
        catalog_id: str,
        text_query: str = Query(None, title='Search terms'),
        facet_query: Json = Query(None, title='Search facets', description='JSON object of facet:value pairs'),
        north_bound: float = Query(None, title='North bound latitude', ge=-90, le=90),
        south_bound: float = Query(None, title='South bound latitude', ge=-90, le=90),
        east_bound: float = Query(None, title='East bound longitude', ge=-180, le=180),
        west_bound: float = Query(None, title='West bound longitude', ge=-180, le=180),
        start_date: date = Query(None, title='Date range start'),
        end_date: date = Query(None, title='Date range end'),
        exclusive_region: bool = Query(False, title='Exclude partial spatial matches'),
        exclusive_interval: bool = Query(False, title='Exclude partial temporal matches'),
        page: int = Query(1, ge=1, title='Page number'),
        size: int = Query(50, ge=0, title='Page size; 0=unlimited'),
        sort: SearchResultSort = Query(SearchResultSort.TIMESTAMP_DESC, title='Sort by'),
):
    if not Session.get(Catalog, catalog_id):
        raise HTTPException(HTTP_404_NOT_FOUND)

    stmt = (
        select(CatalogRecord)
        .where(CatalogRecord.catalog_id == catalog_id)
        .where(CatalogRecord.published)
        .where(CatalogRecord.searchable)
    )

    if text_query and (text_query := text_query.strip()):
        ts_query = func.plainto_tsquery('english', text_query)
        stmt = stmt.where(CatalogRecord.full_text.op('@@')(ts_query))
        if sort == SearchResultSort.RANK_DESC:
            # the third parameter to ts_rank_cd is a normalization bit mask:
            # 1 = divides the rank by 1 + the logarithm of the document length
            # 4 = divides the rank by the mean harmonic distance between extents
            stmt = stmt.add_columns(func.ts_rank_cd(CatalogRecord.full_text, ts_query, 1 | 4).label('rank'))

    if facet_query is not None:
        if not isinstance(facet_query, dict):
            raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'facet_query must be a JSON object')

        for facet_title, facet_value in facet_query.items():
            if not isinstance(facet_value, str):
                raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'facet value must be a string')

            crf = aliased(CatalogRecordFacet, name=f'crf{facet_title}')
            stmt = stmt.join(crf)
            stmt = stmt.where(and_(
                crf.facet == facet_title,
                crf.value == facet_value,
            ))

    if exclusive_region:
        if north_bound is not None:
            stmt = stmt.where(CatalogRecord.spatial_north <= north_bound)

        if south_bound is not None:
            stmt = stmt.where(CatalogRecord.spatial_south >= south_bound)

        if east_bound is not None:
            stmt = stmt.where(CatalogRecord.spatial_east <= east_bound)

        if west_bound is not None:
            stmt = stmt.where(CatalogRecord.spatial_west >= west_bound)

    else:
        if north_bound is not None:
            stmt = stmt.where(CatalogRecord.spatial_south <= north_bound)

        if south_bound is not None:
            stmt = stmt.where(CatalogRecord.spatial_north >= south_bound)

        if east_bound is not None:
            stmt = stmt.where(CatalogRecord.spatial_west <= east_bound)

        if west_bound is not None:
            stmt = stmt.where(CatalogRecord.spatial_east >= west_bound)

    if exclusive_interval:
        if start_date:
            stmt = stmt.where(CatalogRecord.temporal_start >= start_date)

        if end_date:
            stmt = stmt.where(CatalogRecord.temporal_end <= end_date)

    else:
        if start_date:
            stmt = stmt.where(CatalogRecord.temporal_end >= start_date)

        if end_date:
            stmt = stmt.where(CatalogRecord.temporal_start <= end_date)

    total = Session.execute(
        select(func.count())
        .select_from(stmt.subquery())
    ).scalar_one()

    if text_query and sort == SearchResultSort.RANK_DESC:
        order_by = text('rank DESC')
    else:
        order_by = CatalogRecord.timestamp.desc()

    limit = size or total
    items = [
        output_published_record_model(row.CatalogRecord) for row in Session.execute(
            stmt.
            order_by(order_by).
            offset(limit * (page - 1)).
            limit(limit)
        )
    ]

    facets = {}
    facet_subquery = select(CatalogRecordFacet).subquery()
    for row in Session.execute(
        select(
            facet_subquery.c.facet,
            facet_subquery.c.value,
            func.count(),
        )
        .join_from(
            stmt.subquery(),
            facet_subquery,
        )
        .group_by(
            facet_subquery.c.facet,
            facet_subquery.c.value,
        )
    ):
        facets.setdefault(row.facet, [])
        facets[row.facet] += [(row.value, row.count)]

    return SearchResult(
        facets=facets,
        items=items,
        total=total,
        page=page,
        pages=ceil(total / limit) if limit else 0,
    )


async def get_catalog_record_by_id_or_doi(
        catalog_id: str,
        record_id_or_doi: str = Path(..., title='UUID or DOI'),
) -> CatalogRecord:
    """Dependency function for retrieving a published catalog record."""
    stmt = (
        select(CatalogRecord).
        where(CatalogRecord.catalog_id == catalog_id).
        where(CatalogRecord.published)
    )

    try:
        UUID(record_id_or_doi, version=4)
        stmt = stmt.where(CatalogRecord.record_id == record_id_or_doi)

    except ValueError:
        if re.match(DOI_REGEX, record_id_or_doi):
            stmt = stmt.join(Record)
            stmt = stmt.where(func.lower(Record.doi) == record_id_or_doi.lower())
        else:
            raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'Invalid record identifier: expecting a UUID or DOI')

    if not (catalog_record := Session.execute(stmt).scalar_one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return catalog_record


@router.get(
    '/{catalog_id}/records/{record_id_or_doi:path}',
    response_model=PublishedSAEONRecordModel | PublishedDataCiteRecordModel,
    dependencies=[Depends(Authorize(ODPScope.CATALOG_READ))],
)
async def get_record(
        catalog_record: CatalogRecord = Depends(get_catalog_record_by_id_or_doi),
):
    return output_published_record_model(catalog_record)


@router.get(
    '/{catalog_id}/getvalue/{record_id_or_doi:path}',
    response_model=Any,
    dependencies=[Depends(Authorize(ODPScope.CATALOG_READ))],
    description='Get a value from the metadata for a published record',
)
async def get_metadata_value(
        schema_id: str,
        json_pointer: str = Query('', description='JSON pointer reference into the `"metadata"` document selected '
                                                  'from a published record\'s `"metadata_records"` by the given `schema_id`'),
        catalog_record: CatalogRecord = Depends(get_catalog_record_by_id_or_doi),
):
    published_record = output_published_record_model(catalog_record)
    if not isinstance(published_record, PublishedSAEONRecordModel):
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'Function not available for the specified record')

    try:
        metadata_dict = next((
            metadata_record.metadata
            for metadata_record in published_record.metadata_records
            if metadata_record.schema_id == schema_id
        ))
    except StopIteration:
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'Metadata not available for the specified schema')

    try:
        value = JSONPointer(json_pointer).evaluate(metadata_dict)
    except JSONPointerMalformedError as e:
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, str(e))
    except JSONPointerReferenceError:
        return None

    return value


@router.get(
    '/{catalog_id}/external/{record_id}',
    response_model=Optional[dict[str, Any]],
    dependencies=[Depends(Authorize(ODPScope.CATALOG_READ))],
)
async def get_external_record(
        catalog_id: str,
        record_id: str,
        datacite: DataciteClient = Depends(get_datacite_client),
):
    if not Session.get(Catalog, catalog_id):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if catalog_id == ODPCatalog.DATACITE:
        stmt = (
            select(CatalogRecord).
            where(CatalogRecord.catalog_id == catalog_id).
            where(CatalogRecord.record_id == record_id).
            where(CatalogRecord.published)
        )

        if not (catalog_record := Session.execute(stmt).scalar_one_or_none()):
            raise HTTPException(HTTP_404_NOT_FOUND)

        try:
            return datacite.get_doi(catalog_record.record.doi)
        except DataciteError as e:
            raise HTTPException(e.status_code, e.error_detail) from e

    raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'Not an external catalog')


@router.get(
    '/{catalog_id}/go/{record_id_or_doi:path}',
    description='Redirect to the web page for a catalog record.',
)
async def redirect_to(
        catalog_record: CatalogRecord = Depends(get_catalog_record_by_id_or_doi),
):
    url = f'{catalog_record.catalog.url}/'
    url += catalog_record.record.doi if catalog_record.record.doi else catalog_record.record_id

    return RedirectResponse(url)
