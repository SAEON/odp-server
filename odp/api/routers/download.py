from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query, HTTPException, Request, Depends
from starlette.status import HTTP_201_CREATED, HTTP_400_BAD_REQUEST
from odp.api.lib.paging import Page
from odp.api.lib.auth import Authorize
from odp.api.models import DownloadAuditModel, DownloadAuditResponse, DownloadStatsModel
from odp.const import ODPScope
from odp.db.models import DownloadAudit
from odp.lib import download_service

router = APIRouter()


@router.post('/audit', status_code=HTTP_201_CREATED, response_model=DownloadAuditResponse)
async def create_download_audit(request: Request):
    """
    Accept JSON payload to record a download audit.
    Persistence logic follows standard system methods.
    """
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(HTTP_400_BAD_REQUEST, 'Invalid JSON payload')

    meta = payload.get('meta', {}) or {}
    for k in ('name', 'email', 'organisation', 'doi', 'record_id', 'catalog_url'):
        if payload.get(k) is not None:
            meta[k] = payload.get(k)

    audit = DownloadAudit(
        client_id=payload.get('client_id') or 'unknown',
        user_id=payload.get('user_id'),
        download_url=payload.get('download_url'),
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get('user-agent'),
        file_size=payload.get('file_size'),
        success=bool(payload.get('success', True)),
        timestamp=datetime.now(timezone.utc),
        meta=meta,
    )
    audit.save()
    return {"status": "ok", "audit_id": audit.id}


@router.get('/logs',
            response_model=Page[DownloadAuditModel],
            dependencies=[Depends(Authorize(ODPScope.CATALOG_READ))],
            )
async def get_download_logs(
        start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
        end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
        email: Optional[str] = None,
        name: Optional[str] = None,
        organisation: Optional[str] = None,
        download_type: Optional[str] = None,
        page: int = Query(1, ge=1),
        size: int = Query(50, ge=1, le=200),
):
    """
    Delegates all filtering and pagination logic to the service.
    """
    return download_service.get_download_logs(
        start_date=start_date,
        end_date=end_date,
        email=email,
        name=name,
        organisation=organisation,
        download_type=download_type,
        page=page,
        size=size
    )


@router.get('/stats',
            response_model=DownloadStatsModel,
            dependencies=[Depends(Authorize(ODPScope.CATALOG_READ))],
            )
async def get_download_statistics(
        start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
        end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
):
    """
    Delegates statistics calculation to the service.
    """
    return download_service.get_download_stats(
        start_date=start_date,
        end_date=end_date
    )


@router.get('/export/csv',
            dependencies=[Depends(Authorize(ODPScope.CATALOG_READ))],
            )
async def export_downloads_csv(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        email: Optional[str] = None,
        organisation: Optional[str] = None,
        download_type: Optional[str] = None,
):
    """
    Delegates CSV generation to the service.
    """
    return download_service.generate_downloads_csv(
        start_date=start_date,
        end_date=end_date,
        email=email,
        organisation=organisation,
        download_type=download_type,
    )
