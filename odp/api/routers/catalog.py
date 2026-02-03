import re
from datetime import date, datetime, timezone
from enum import Enum
from functools import partial
from io import BytesIO
from math import ceil
from typing import Any, Optional, List
from uuid import UUID
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from jschon import JSONPointer
from jschon.exc import JSONPointerMalformedError, JSONPointerReferenceError
from pydantic import Json
import requests

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch

from sqlalchemy import func

from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.orm import aliased, load_only
from starlette.status import HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_ENTITY

from odp.api.lib.auth import Authorize
from odp.api.lib.datacite import get_datacite_client
from odp.api.lib.paging import Page, Paginator
from odp.api.lib.utils import output_published_record_model
from odp.api.models import (CatalogModel, CatalogModelWithData, PublishedDataCiteRecordModel, PublishedSAEONRecordModel, RetractedRecordModel,
                            SearchResult)
from odp.const import DOI_REGEX, ODPCatalog, ODPScope
from odp.db import Session
from odp.db.models import Catalog, CatalogRecord, CatalogRecordFacet, PublishedRecord, Record, DownloadAudit
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
        stmt = stmt.add_columns(func.plainto_tsquery('english', text_query).column_valued('query'))
        stmt = stmt.where(text('full_text @@ query'))
        if sort == SearchResultSort.RANK_DESC:
            # the third parameter to ts_rank_cd is a normalization bit mask:
            # 1 = divides the rank by 1 + the logarithm of the document length
            # 4 = divides the rank by the mean harmonic distance between extents
            stmt = stmt.add_columns(func.ts_rank_cd('full_text', 'query', 1 | 4).label('rank'))

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


@router.get(
    '/{catalog_id}/subset',
    response_model=SearchResult,
    dependencies=[Depends(Authorize(ODPScope.CATALOG_SEARCH))],
    description="Return a catalog's subset published records.",
)
async def records_subset(
        catalog_id: str,
        record_id_or_doi_list: List[str] = Query(..., alias="record_id_or_doi_list"),
        page: int = 1,
        size: int = 50
):

    if not Session.get(Catalog, catalog_id):
        raise HTTPException(HTTP_404_NOT_FOUND)

    stmt = (
        select(CatalogRecord)
        .where(CatalogRecord.catalog_id == catalog_id)
        .where(CatalogRecord.record_id.in_(record_id_or_doi_list))
        .where(CatalogRecord.published)
        .where(CatalogRecord.searchable)
    )



    total = Session.execute(
        select(func.count())
        .select_from(stmt.subquery())
    ).scalar_one()


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

@router.post('/download/bundle')
async def create_download_bundle(
        request: Request,
        catalog_id: str = Query(...),
        record_dois: List[str] = Query(...),
):
    """
    Create a ZIP bundle containing metadata PDFs for multiple records.

    This endpoint:
    1. Validates that all records exist in the catalog
    2. Fetches metadata for each record
    3. Generates metadata PDF for each record
    4. Creates a ZIP archive containing everything
    5. Streams the ZIP back to the client
    6. Logs the download to the download_audit table

    Query Parameters:
    - catalog_id: The catalog identifier (e.g., 'mims')
    - record_dois: List of DOI strings to bundle

    Request Body (JSON):
    {
        "user_metadata": {
            "name": "User Name",
            "email": "user@example.com",
            "organisation": "Organisation Name",
            "reason": "Research purposes"
        }
    }

    Returns:
    - StreamingResponse with ZIP file
    """
    # Parse request body
    try:
        body = await request.json()
        user_metadata = body.get('user_metadata', {})
    except Exception as e:
        raise HTTPException(400, f'Invalid JSON payload: {str(e)}')

    # Validate inputs
    if not record_dois:
        raise HTTPException(400, 'record_dois parameter is required')
    if not user_metadata.get('email'):
        raise HTTPException(400, 'user_metadata.email is required')

    # Maximum 2 GB per bundle
    MAX_BUNDLE_SIZE = 2 * 1024 * 1024 * 1024

    try:
        # Validate that all records exist
        with Session() as session:
            for doi in record_dois:
                record = session.query(CatalogRecord).filter(
                    CatalogRecord.catalog_id == catalog_id,
                    CatalogRecord.doi == doi
                ).first()
                if not record:
                    raise HTTPException(404, f'Record {doi} not found')

        # Create ZIP buffer
        zip_buffer = BytesIO()
        total_size = 0
        processed_records = []
        files_added = []  # Track files added to ZIP

        with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zip_file:
            for doi in record_dois:
                try:
                    # Fetch record metadata
                    with Session() as session:
                        catalog_record = session.query(CatalogRecord).filter(
                            CatalogRecord.catalog_id == catalog_id,
                            CatalogRecord.doi == doi
                        ).first()

                        if not catalog_record:
                            continue

                        # Safe folder name from DOI
                        record_title = doi.replace('/', '_')[:50]

                        # Get record data
                        record_data = catalog_record.record.to_dict() if hasattr(catalog_record, 'record') else {}

                    # Generate metadata PDF using unified module
                    try:
                        from odp.lib.metadata_adapters import adapt_metadata
                        from odp.lib.metadata_pdf import generate_pdf

                        if not record_data:
                            print(f'Warning: Empty record data for {doi}')
                            continue

                        # Extract metadata and adapt to unified format
                        metadata = record_data.get('metadata', {})
                        keywords = record_data.get('keywords', [])

                        try:
                            # Adapt to unified RecordMetadata format
                            record_metadata = adapt_metadata(metadata)
                            if keywords:
                                record_metadata.keywords = keywords

                            # Generate PDF from unified format
                            pdf_buffer = generate_pdf(record_metadata)
                        except (ValueError, KeyError) as adapt_err:
                            # Fallback to legacy function for backward compatibility
                            print(f'Info: Falling back to legacy PDF generation for {doi}: {str(adapt_err)}')
                            pdf_buffer = build_metadata_pdf(record_data)

                        pdf_blob = pdf_buffer.getvalue()

                        if not pdf_blob:
                            print(f'Warning: Generated empty PDF for {doi}')
                            continue

                        folder_name = record_title
                        pdf_filename = f'{folder_name}/metadata.pdf'
                        pdf_size = len(pdf_blob)

                        zip_file.writestr(pdf_filename, pdf_blob)
                        total_size += pdf_size
                        processed_records.append(doi)

                        # Track file information
                        files_added.append({
                            'name': pdf_filename,
                            'size': pdf_size,
                            'doi': doi,
                            'type': 'metadata_pdf'
                        })

                        print(f'Debug: Added {doi} to ZIP ({pdf_size} bytes)')
                    except Exception as pdf_err:
                        print(f'Warning: Could not generate PDF for {doi}: {str(pdf_err)}')
                        continue

                    # Check size limit
                    if total_size > MAX_BUNDLE_SIZE:
                        raise HTTPException(413, 'Bundle exceeds maximum size of 2 GB')

                except HTTPException:
                    raise
                except Exception as e:
                    print(f'Error processing record {doi}: {str(e)}')
                    continue

        # ZipFile context is closed, get final ZIP size
        # The buffer now contains the complete ZIP file
        zip_contents = zip_buffer.getvalue()
        final_size = len(zip_contents)
        zip_buffer.seek(0)

        # Log to download_audit
        try:
            # Ensure we have a valid file size before logging
            if final_size <= 0:
                print(f'Warning: ZIP buffer is empty (size: {final_size}), processed {len(processed_records)} records')

            with Session() as session:
                audit = DownloadAudit(
                    client_id='mims-client',
                    user_id=None,
                    download_url=f'/catalog/download/bundle?catalog_id={catalog_id}',
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get('user-agent'),
                    file_size=final_size if final_size > 0 else None,
                    success=True,
                    timestamp=datetime.now(timezone.utc),
                    meta={
                        'name': user_metadata.get('name', 'N/A'),
                        'email': user_metadata.get('email', 'N/A'),
                        'organisation': user_metadata.get('organisation', 'N/A'),
                        'download_type': 'zip_bundle',
                        'record_count': len(processed_records),
                        'dois': processed_records,
                        'reason': user_metadata.get('reason', 'N/A'),
                        'source': 'MIMS-UI-Bundle',
                        'bundle_size_bytes': final_size,
                        'files_in_bundle': files_added,
                        'total_files': len(files_added),
                        'zip_file_size': final_size,
                    }
                )
                session.add(audit)
                session.commit()
        except Exception as audit_err:
            print(f'Warning: Could not log to download_audit: {str(audit_err)}')

        # Return as streaming response
        return StreamingResponse(
            iter([zip_buffer.getvalue()]),
            media_type='application/zip',
            headers={
                'Content-Disposition': 'attachment; filename="records.zip"',
                'Content-Length': str(final_size)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f'Error creating download bundle: {str(e)}')
        raise HTTPException(500, f'Error creating bundle: {str(e)}')


# ============================================================================
# PDF Generation API Endpoints (Unified Module)
# ============================================================================

@router.post(
    '/metadata/generate-pdf',
    response_class=StreamingResponse,
    summary='Generate PDF from metadata',
    description='Generate a PDF from metadata in DataCite4 or ISO19115 format',
    include_in_schema=True,
)
async def generate_metadata_pdf(request: Request):
    """
    Generate PDF from raw metadata.

    Accepts metadata in either DataCite4 or ISO19115 format.
    Automatically detects format or uses specified schema_id.

    Request body:
    {
        "metadata_format": "auto|datacite4|iso19115",
        "metadata": {...},
        "keywords": [...],
        "temporal_start": "ISO 8601 date",
        "temporal_end": "ISO 8601 date"
    }
    """
    try:
        from odp.lib.metadata_adapters import adapt_metadata
        from odp.lib.metadata_pdf import generate_pdf

        body = await request.json()

        # Get parameters
        metadata = body.get('metadata')
        metadata_format = body.get('metadata_format', 'auto')
        keywords = body.get('keywords', [])
        temporal_start = body.get('temporal_start')
        temporal_end = body.get('temporal_end')

        if not metadata:
            raise HTTPException(400, 'metadata field is required')

        # Adapt metadata to unified format
        try:
            record_metadata = adapt_metadata(metadata, schema_id=metadata_format)
        except ValueError as e:
            raise HTTPException(422, f'Could not process metadata: {str(e)}')

        # Override with request parameters if provided
        if keywords:
            record_metadata.keywords = keywords
        if temporal_start:
            from odp.lib.metadata_pdf import TemporalExtent
            if temporal_end:
                record_metadata.temporal = TemporalExtent(
                    start_date=temporal_start,
                    end_date=temporal_end
                )

        # Generate PDF
        try:
            pdf_buffer = generate_pdf(record_metadata)
            pdf_content = pdf_buffer.getvalue()
        except ValueError as e:
            raise HTTPException(500, f'PDF generation failed: {str(e)}')

        # Note: PDF generation is typically called internally by other endpoints
        # (e.g., MIMS downloads or ZIP bundle generation) which handle their own
        # audit logging. We don't log here to avoid duplicate audit entries.

        return StreamingResponse(
            iter([pdf_content]),
            media_type='application/pdf',
            headers={
                'Content-Disposition': 'attachment; filename="metadata.pdf"',
                'Content-Length': str(len(pdf_content))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f'Error generating PDF: {str(e)}')
        raise HTTPException(500, f'Internal server error: {str(e)}')


@router.post(
    '/{catalog_id}/records/{record_id}/metadata.pdf',
    response_class=StreamingResponse,
    summary='Generate PDF for specific record',
    description='Generate PDF for a catalog record by ID',
)
async def generate_record_pdf(
    catalog_id: str,
    record_id: UUID,
    request: Request,
):
    """
    Generate PDF for a specific catalog record.

    Retrieves record metadata and generates PDF directly.
    """
    try:
        from odp.lib.metadata_adapters import adapt_metadata
        from odp.lib.metadata_pdf import generate_pdf

        # Fetch record from database
        stmt = select(CatalogRecord).where(
            and_(
                CatalogRecord.catalog_id == catalog_id,
                CatalogRecord.record_id == record_id,
                CatalogRecord.published == True
            )
        )

        if not (catalog_record := Session.execute(stmt).scalar_one_or_none()):
            raise HTTPException(404, 'Record not found')

        # Extract metadata
        metadata = catalog_record.record.data.get('metadata', {})
        keywords = catalog_record.record.data.get('keywords', [])

        # Adapt to unified format
        try:
            record_metadata = adapt_metadata(metadata)
            if keywords:
                record_metadata.keywords = keywords
        except ValueError as e:
            raise HTTPException(422, f'Could not process metadata: {str(e)}')

        # Generate PDF
        try:
            pdf_buffer = generate_pdf(record_metadata)
            pdf_content = pdf_buffer.getvalue()
        except ValueError as e:
            raise HTTPException(500, f'PDF generation failed: {str(e)}')

        # Note: PDF generation is typically called internally by other endpoints
        # which handle their own audit logging. We don't log here to avoid
        # duplicate or incomplete audit entries.

        return StreamingResponse(
            iter([pdf_content]),
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="record_{record_id}.pdf"',
                'Content-Length': str(len(pdf_content))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f'Error generating record PDF: {str(e)}')
        raise HTTPException(500, f'Internal server error: {str(e)}')


# ============================================================================
# PDF Generation Utility (Legacy - Kept for backward compatibility)
# ============================================================================

def build_metadata_pdf(record_data: dict) -> BytesIO:
    """
    Generate a metadata PDF for a single catalog record.
    
    Args:
        record_data: Dictionary containing record metadata with structure:
                    {
                        "metadata_records": [{
                            "metadata": {...datacite fields...}
                        }],
                        "keywords": [...],
                        "temporal_start": "ISO timestamp",
                        "temporal_end": "ISO timestamp"
                    }
    
    Returns:
        BytesIO buffer containing the PDF
    """
    try:
        # Extract metadata
        meta = record_data.get("metadata_records", [{}])[0].get("metadata", {})
        
        # Helper function to extract person details
        def _get_person(person):
            """Extract name, affiliation, email, and ORCID from person dict."""
            name = person.get("name", "N/A")
            affiliation = "N/A"
            email = "N/A"
            orcid = "N/A"

            for aff in person.get("affiliation", []):
                if "email:" in aff.get("affiliation", ""):
                    affiliation, email = map(str.strip, aff["affiliation"].split(", email:"))
                else:
                    affiliation = aff.get("affiliation", "N/A")

            for idf in person.get("nameIdentifiers", []):
                if idf.get("nameIdentifierScheme") == "ORCID":
                    orcid = idf.get("nameIdentifier", "N/A")

            return name, affiliation, email, orcid

        # Extract high-level fields
        title = meta.get("titles", [{}])[0].get("title", "Untitled")
        doi = meta.get("doi", "N/A")
        publisher = meta.get("publisher", "N/A")
        pub_year = meta.get("publicationYear", "N/A")
        keywords = ", ".join(record_data.get("keywords", []))

        abstract = meta.get("descriptions", [{}])[0].get("description", "N/A")
        
        # Format temporal extent
        try:
            t_start = datetime.fromisoformat(record_data.get("temporal_start", "")).strftime("%d %b %Y")
            t_end = datetime.fromisoformat(record_data.get("temporal_end", "")).strftime("%d %b %Y")
        except (ValueError, AttributeError):
            t_start = record_data.get("temporal_start", "N/A")
            t_end = record_data.get("temporal_end", "N/A")

        # Geographic extent
        try:
            geo_box = meta.get("geoLocations", [{}])[0].get("geoLocationBox", {})
            geo_str = (
                f"North: {geo_box.get('northBoundLatitude', 'N/A')}\n"
                f"South: {geo_box.get('southBoundLatitude', 'N/A')}\n"
                f"West: {geo_box.get('westBoundLongitude', 'N/A')}\n"
                f"East: {geo_box.get('eastBoundLongitude', 'N/A')}"
            )
        except (IndexError, KeyError):
            geo_str = "N/A"

        # Creator and contributor info
        creator = meta.get("creators", [{}])[0]
        contributor = meta.get("contributors", [{}])[0]
        cr_name, cr_aff, cr_email, cr_orcid = _get_person(creator)
        c_name, c_aff, c_email, c_orcid = _get_person(contributor)

        # License info
        try:
            licence = meta.get("rightsList", [{}])[0]
            licence_txt = f'<link href="{licence.get("rightsURI", "#")}">{licence.get("rights", "N/A")}</link>'
        except (IndexError, KeyError):
            licence_txt = "N/A"

        # Setup styles
        styles = getSampleStyleSheet()
        label_style = ParagraphStyle(
            "label",
            parent=styles["BodyText"],
            fontSize=10,
            leading=13,
            spaceAfter=0,
            spaceBefore=2,
            leftIndent=0,
            rightIndent=6,
            textColor=colors.black,
            wordWrap="LTR",
            bold=True,
        )
        value_style = ParagraphStyle(
            "value",
            parent=styles["BodyText"],
            fontSize=10,
            leading=13,
            spaceAfter=0,
            spaceBefore=2,
        )
        title_value_style = ParagraphStyle(
            "title_value",
            parent=value_style,
            fontSize=11,
            leading=14,
            spaceBefore=0,
            spaceAfter=2,
            bold=True,
        )

        # Build table rows
        rows = [
            [Paragraph("Title", label_style), Paragraph(title, title_value_style)],
            [Paragraph("DOI", label_style), Paragraph(f'<link href="https://doi.org/{doi}">https://doi.org/{doi}</link>', value_style)],
            [Paragraph("Authors", label_style), Paragraph(f"{cr_name}<br/>{cr_aff}, email: {cr_email}", value_style)],
            [Paragraph("Publisher", label_style), Paragraph(f"{publisher} ({pub_year})", value_style)],
            [Paragraph("Contributors", label_style), Paragraph(f"Contact Person: {c_name}<br/>{c_aff},<br/>email: {c_email}", value_style)],
            [Paragraph("Abstract", label_style), Paragraph(abstract, value_style)],
            [Paragraph("Data", label_style), Paragraph(licence_txt, value_style)],
            [Paragraph("Temporal extent", label_style), Paragraph(f"{t_start} – {t_end}", value_style)],
            [Paragraph("Geographic extent", label_style), Paragraph(geo_str.replace("\n", "<br/>"), value_style)],
            [Paragraph("Keywords", label_style), Paragraph(keywords, value_style)],
        ]

        # Create table
        table = Table(
            rows,
            colWidths=[1.6 * inch, 5.3 * inch],
            hAlign="LEFT",
            repeatRows=0,
        )

        # Table styling
        tbl_style = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, 0), 0.25, colors.lightgrey),
        ]
        for r in range(1, len(rows)):
            tbl_style.append(("LINEBELOW", (0, r), (-1, r), 0.25, colors.lightgrey))

        table.setStyle(TableStyle(tbl_style))

        # Build PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        story = [table, Spacer(1, 0.2 * inch)]
        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        raise
