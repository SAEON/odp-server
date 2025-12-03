from datetime import datetime, timezone
from io import StringIO
import csv
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_201_CREATED, HTTP_400_BAD_REQUEST
from sqlalchemy import insert, desc, func

from odp.db import Session
from odp.db.models import DownloadAudit
from odp.api.lib.auth import Authorize, Authorized  # only if you want auth; otherwise omit

router = APIRouter()

@router.post('/audit', status_code=HTTP_201_CREATED)
async def create_download_audit(request: Request):
    """
    Accept JSON payload to record a download audit and return success.

    Expected payload fields:
    - download_url (required): URL of the downloaded file
    - name (optional): User's name
    - email (optional): User's email address
    - organisation (optional): User's organisation
    - doi (optional): DOI of the record being downloaded
    - record_id (optional): Record ID of the record being downloaded
    - file_size (optional): Size of the downloaded file in bytes
    - success (optional): Whether download was successful (default: true)
    - client_id (optional): Client identifier (derived from request if not provided)
    - user_id (optional): User identifier (derived from request if not provided)
    - meta (optional): Additional metadata object to store
    """
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(HTTP_400_BAD_REQUEST, 'Invalid JSON payload')

    # derive client_id/user_id from request context (if you have that info).
    # Use placeholders if not available
    client_id = getattr(request.state, 'client_id', None) or payload.get('client_id') or 'unknown'
    user_id = getattr(request.state, 'user_id', None) or payload.get('user_id')

    download_url = payload.get('download_url')
    file_size = payload.get('file_size')
    success = bool(payload.get('success', True))
    meta = payload.get('meta', {}) or {}

    # copy optional form fields into meta for storage
    for k in ('name', 'email', 'organisation', 'doi', 'record_id'):
        if payload.get(k) is not None:
            meta[k] = payload.get(k)

    # capture IP and UA
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get('user-agent')

    # persist using Session
    with Session() as session:
        audit = DownloadAudit(
            client_id=client_id,
            user_id=user_id,
            download_url=download_url,
            ip_address=ip_address,
            user_agent=user_agent,
            file_size=file_size,
            success=success,
            timestamp=datetime.now(timezone.utc),
            meta=meta,
        )
        session.add(audit)
        session.commit()
        audit_id = audit.id

    return {"status": "ok", "audit_id": audit_id}


@router.get('/logs')
async def get_download_logs(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    email: Optional[str] = None,
    organisation: Optional[str] = None,
    download_type: Optional[str] = None,
    page: int = 1,
    size: int = 50,
):
    """
    Get paginated download audit logs with optional filtering.

    Query parameters:
    - start_date: ISO format date string (YYYY-MM-DD)
    - end_date: ISO format date string (YYYY-MM-DD)
    - email: Filter by user email
    - organisation: Filter by organisation
    - download_type: Filter by download_type in meta (e.g., 'single_record', 'zip_bundle')
    - page: Page number (default 1)
    - size: Page size (default 50, max 200)
    """

    # Validate page size
    size = min(size, 200)
    offset = (page - 1) * size

    with Session() as session:
        query = session.query(DownloadAudit)

        # Apply filters
        if start_date:
            try:
                start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
                query = query.filter(DownloadAudit.timestamp >= start)
            except ValueError:
                raise HTTPException(HTTP_400_BAD_REQUEST, 'Invalid start_date format (use YYYY-MM-DD)')

        if end_date:
            try:
                end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
                # Add 1 day to include the entire end date
                end = end.replace(hour=23, minute=59, second=59)
                query = query.filter(DownloadAudit.timestamp <= end)
            except ValueError:
                raise HTTPException(HTTP_400_BAD_REQUEST, 'Invalid end_date format (use YYYY-MM-DD)')

        if email:
            query = query.filter(DownloadAudit.meta['email'].astext == email)

        if organisation:
            query = query.filter(DownloadAudit.meta['organisation'].astext == organisation)

        if download_type:
            query = query.filter(DownloadAudit.meta['download_type'].astext == download_type)

        # Get total count
        total = query.count()

        # Order by timestamp descending and paginate
        downloads = query.order_by(desc(DownloadAudit.timestamp)).offset(offset).limit(size).all()

        # Convert to dictionaries
        result = []
        for d in downloads:
            result.append({
                'id': d.id,
                'client_id': d.client_id,
                'timestamp': d.timestamp.isoformat(),
                'email': d.meta.get('email') if d.meta else None,
                'name': d.meta.get('name') if d.meta else None,
                'organisation': d.meta.get('organisation') if d.meta else None,
                'download_type': d.meta.get('download_type') if d.meta else None,
                'file_size': d.file_size,
                'success': d.success,
                'ip_address': d.ip_address,
                'meta': d.meta or {},
            })

        return {
            'total': total,
            'page': page,
            'size': size,
            'total_pages': (total + size - 1) // size,
            'items': result,
        }


@router.get('/stats')
async def get_download_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Get download statistics and aggregated data.

    Query parameters:
    - start_date: ISO format date string (YYYY-MM-DD)
    - end_date: ISO format date string (YYYY-MM-DD)
    """

    with Session() as session:
        query = session.query(DownloadAudit)

        # Apply date filters
        if start_date:
            try:
                start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
                query = query.filter(DownloadAudit.timestamp >= start)
            except ValueError:
                raise HTTPException(HTTP_400_BAD_REQUEST, 'Invalid start_date format (use YYYY-MM-DD)')

        if end_date:
            try:
                end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
                end = end.replace(hour=23, minute=59, second=59)
                query = query.filter(DownloadAudit.timestamp <= end)
            except ValueError:
                raise HTTPException(HTTP_400_BAD_REQUEST, 'Invalid end_date format (use YYYY-MM-DD)')

        # Total downloads
        total_downloads = query.count()

        # Total unique users (by email in meta)
        unique_users = session.query(func.count(func.distinct(DownloadAudit.meta['email']))).filter(
            DownloadAudit.meta['email'].isnot(None)
        )
        if start_date:
            unique_users = unique_users.filter(DownloadAudit.timestamp >= datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc))
        if end_date:
            unique_users = unique_users.filter(DownloadAudit.timestamp <= datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).replace(hour=23, minute=59, second=59))
        unique_users_count = unique_users.scalar() or 0

        # Total data volume (sum of file_size)
        total_volume = query.filter(DownloadAudit.file_size.isnot(None)).with_entities(func.sum(DownloadAudit.file_size)).scalar() or 0

        # Successful downloads
        successful_downloads = query.filter(DownloadAudit.success == True).count()

        # Failed downloads
        failed_downloads = query.filter(DownloadAudit.success == False).count()

        # Downloads by type
        downloads_by_type = session.query(
            DownloadAudit.meta['download_type'].astext.label('type'),
            func.count(DownloadAudit.id).label('count')
        ).filter(DownloadAudit.meta['download_type'].isnot(None))

        if start_date:
            downloads_by_type = downloads_by_type.filter(DownloadAudit.timestamp >= datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc))
        if end_date:
            downloads_by_type = downloads_by_type.filter(DownloadAudit.timestamp <= datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).replace(hour=23, minute=59, second=59))

        downloads_by_type = downloads_by_type.group_by('type').all()

        # Downloads by organisation (top 20)
        organisations = session.query(
            DownloadAudit.meta['organisation'].astext.label('organisation'),
            func.count(DownloadAudit.id).label('downloads'),
            func.count(func.distinct(DownloadAudit.meta['email'])).label('unique_users')
        ).filter(DownloadAudit.meta['organisation'].isnot(None))

        if start_date:
            organisations = organisations.filter(DownloadAudit.timestamp >= datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc))
        if end_date:
            organisations = organisations.filter(DownloadAudit.timestamp <= datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).replace(hour=23, minute=59, second=59))

        organisations = organisations.group_by('organisation').order_by(func.count(DownloadAudit.id).desc()).limit(20).all()

        # Top 10 most downloaded records (by DOI)
        top_records = session.query(
            DownloadAudit.meta['doi'].astext.label('doi'),
            DownloadAudit.meta['record_id'].astext.label('record_id'),
            func.count(DownloadAudit.id).label('downloads'),
            func.count(func.distinct(DownloadAudit.meta['email'])).label('unique_users')
        ).filter(DownloadAudit.meta['doi'].isnot(None))

        if start_date:
            top_records = top_records.filter(DownloadAudit.timestamp >= datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc))
        if end_date:
            top_records = top_records.filter(DownloadAudit.timestamp <= datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).replace(hour=23, minute=59, second=59))

        top_records = top_records.group_by('doi', 'record_id').order_by(func.count(DownloadAudit.id).desc()).limit(10).all()

        # Daily downloads time-series
        daily_downloads = session.query(
            func.date(DownloadAudit.timestamp).label('date'),
            func.count(DownloadAudit.id).label('downloads'),
            func.count(func.filter(DownloadAudit.success == True, DownloadAudit.id)).label('successful'),
            func.count(func.filter(DownloadAudit.success == False, DownloadAudit.id)).label('failed')
        )

        if start_date:
            daily_downloads = daily_downloads.filter(DownloadAudit.timestamp >= datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc))
        if end_date:
            daily_downloads = daily_downloads.filter(DownloadAudit.timestamp <= datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).replace(hour=23, minute=59, second=59))

        daily_downloads = daily_downloads.group_by(func.date(DownloadAudit.timestamp)).order_by(func.date(DownloadAudit.timestamp)).all()

        return {
            'total_downloads': total_downloads,
            'unique_users': unique_users_count,
            'total_data_volume': total_volume,
            'successful_downloads': successful_downloads,
            'failed_downloads': failed_downloads,
            'downloads_by_type': {
                item[0]: item[1] for item in downloads_by_type
            },
            'organisations': [
                {
                    'name': item[0] or 'Unknown',
                    'downloads': item[1],
                    'unique_users': item[2]
                }
                for item in organisations
            ],
            'top_records': [
                {
                    'doi': item[0],
                    'record_id': item[1],
                    'downloads': item[2],
                    'unique_users': item[3]
                }
                for item in top_records
            ],
            'daily_downloads': [
                {
                    'date': item[0].isoformat() if item[0] else None,
                    'downloads': item[1],
                    'successful': item[2],
                    'failed': item[3]
                }
                for item in daily_downloads
            ],
        }


@router.get('/export/csv')
async def export_downloads_csv(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    email: Optional[str] = None,
    organisation: Optional[str] = None,
    download_type: Optional[str] = None,
):
    """
    Export download logs as CSV file.

    Query parameters:
    - start_date: ISO format date string (YYYY-MM-DD)
    - end_date: ISO format date string (YYYY-MM-DD)
    - email: Filter by user email
    - organisation: Filter by organisation
    - download_type: Filter by download_type in meta
    """

    with Session() as session:
        query = session.query(DownloadAudit)

        # Apply filters
        if start_date:
            try:
                start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
                query = query.filter(DownloadAudit.timestamp >= start)
            except ValueError:
                raise HTTPException(HTTP_400_BAD_REQUEST, 'Invalid start_date format (use YYYY-MM-DD)')

        if end_date:
            try:
                end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
                end = end.replace(hour=23, minute=59, second=59)
                query = query.filter(DownloadAudit.timestamp <= end)
            except ValueError:
                raise HTTPException(HTTP_400_BAD_REQUEST, 'Invalid end_date format (use YYYY-MM-DD)')

        if email:
            query = query.filter(DownloadAudit.meta['email'].astext == email)

        if organisation:
            query = query.filter(DownloadAudit.meta['organisation'].astext == organisation)

        if download_type:
            query = query.filter(DownloadAudit.meta['download_type'].astext == download_type)

        # Order by timestamp descending
        downloads = query.order_by(desc(DownloadAudit.timestamp)).all()

        # Generate CSV
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'ID',
            'Timestamp',
            'Name',
            'Email',
            'Organisation',
            'Download Type',
            'File Size (bytes)',
            'Success',
            'IP Address',
            'Download URL',
        ])

        # Write data rows
        for d in downloads:
            writer.writerow([
                d.id,
                d.timestamp.isoformat(),
                d.meta.get('name') if d.meta else '',
                d.meta.get('email') if d.meta else '',
                d.meta.get('organisation') if d.meta else '',
                d.meta.get('download_type') if d.meta else '',
                d.file_size or '',
                'Yes' if d.success else 'No',
                d.ip_address or '',
                d.download_url or '',
            ])

        # Generate filename with date range
        filename = 'download_logs'
        if start_date and end_date:
            filename += f'_{start_date}_to_{end_date}'
        elif start_date:
            filename += f'_{start_date}'
        filename += '.csv'

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'},
        )