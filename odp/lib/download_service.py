import csv
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, Optional

from fastapi import HTTPException
from sqlalchemy import desc, func, case
from starlette.responses import StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST

from odp.db import Session
from odp.db.models import DownloadAudit


def _apply_filters(query, start_date, end_date, name, email, organisation, download_type):
    """Internal helper to standardize filtering across all service methods."""
    try:
        if start_date:
            query = query.filter(
                DownloadAudit.timestamp >= datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc))
        if end_date:
            query = query.filter(
                DownloadAudit.timestamp <= datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).replace(
                    hour=23, minute=59, second=59))
    except ValueError:
        raise HTTPException(HTTP_400_BAD_REQUEST, "Invalid date format. Use YYYY-MM-DD")

    if email:
        query = query.filter(DownloadAudit.meta['email'].astext == email)
    if name:
        query = query.filter(DownloadAudit.meta['name'].astext == email)
    if organisation:
        query = query.filter(DownloadAudit.meta['organisation'].astext == organisation)
    if download_type:
        query = query.filter(DownloadAudit.meta['download_type'].astext == download_type)
    return query


def get_download_stats(start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    with Session() as session:
        query = _apply_filters(session.query(DownloadAudit), start_date, end_date, None, None, None, None)

        total_downloads = query.count()
        total_volume = query.with_entities(func.sum(DownloadAudit.file_size)).scalar() or 0
        successful_downloads = query.filter(DownloadAudit.success == True).count()
        failed_downloads = query.filter(DownloadAudit.success == False).count()

        unique_users_count = session.query(func.count(func.distinct(DownloadAudit.meta['email']))).filter(
            DownloadAudit.meta['email'].isnot(None)
        ).scalar() or 0

        downloads_by_type = session.query(
            DownloadAudit.meta['download_type'].astext.label('type'),
            func.count(DownloadAudit.id).label('count')
        ).group_by('type').all()

        organisations = session.query(
            DownloadAudit.meta['organisation'].astext.label('organisation'),
            func.count(DownloadAudit.id).label('downloads')
        ).filter(DownloadAudit.meta['organisation'].isnot(None)) \
            .group_by('organisation').order_by(desc('downloads')).limit(20).all()

        # Top 10 most downloaded records (by DOI)
        top_records = session.query(
            DownloadAudit.meta['doi'].astext.label('doi'),
            DownloadAudit.meta['record_id'].astext.label('record_id'),
            func.count(DownloadAudit.id).label('downloads'),
            func.count(func.distinct(DownloadAudit.meta['email'])).label('unique_users')
        ).filter(DownloadAudit.meta['doi'].isnot(None))

        if start_date:
            top_records = top_records.filter(
                DownloadAudit.timestamp >= datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc))
        if end_date:
            top_records = top_records.filter(
                DownloadAudit.timestamp <= datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).replace(
                    hour=23, minute=59, second=59))

        top_records = top_records.group_by('doi', 'record_id').order_by(func.count(DownloadAudit.id).desc()).limit(
            10).all()

        # Daily downloads time-series
        daily_downloads = session.query(
            func.date(DownloadAudit.timestamp).label('date'),
            func.count(DownloadAudit.id).label('downloads'),
            func.sum(case((DownloadAudit.success == True, 1), else_=0)).label('successful'),
            func.sum(case((DownloadAudit.success == False, 1), else_=0)).label('failed')
        )

        if start_date:
            daily_downloads = daily_downloads.filter(
                DownloadAudit.timestamp >= datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc))
        if end_date:
            daily_downloads = daily_downloads.filter(
                DownloadAudit.timestamp <= datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).replace(
                    hour=23, minute=59, second=59))

        daily_downloads = daily_downloads.group_by(func.date(DownloadAudit.timestamp)).order_by(
            func.date(DownloadAudit.timestamp)).all()

        return {
            'total_downloads': total_downloads,
            'unique_users': unique_users_count,
            'total_data_volume': total_volume,
            'successful_downloads': successful_downloads,
            'failed_downloads': failed_downloads,
            'downloads_by_type': {item.type: item.count for item in downloads_by_type if item.type},
            'organisations': [{'name': item.organisation, 'downloads': item.downloads} for item in organisations],
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


def get_download_logs(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        email: Optional[str] = None,
        name: Optional[str] = None,
        organisation: Optional[str] = None,
        download_type: Optional[str] = None,
        page: int = 1,
        size: int = 50,
) -> Dict[str, Any]:
    offset = (page - 1) * size
    with Session() as session:
        query = _apply_filters(session.query(DownloadAudit), start_date, end_date, name, email, organisation,
                               download_type)
        total = query.count()
        downloads = query.order_by(desc(DownloadAudit.timestamp)).offset(offset).limit(size).all()

        items = [{
            'id': d.id,
            'timestamp': d.timestamp.isoformat(),
            'name': d.meta.get('name'),
            'email': d.meta.get('email') if d.meta else None,
            'organisation': d.meta.get('organisation') if d.meta else None,
            'download_type': d.meta.get('download_type') if d.meta else None,
            'success': d.success,
            'ip_address': d.ip_address,
            'doi': d.meta.get('doi') if d.meta else None,
            'record_ids': d.meta.get('record_ids', []) if d.meta else [],
        } for d in downloads]

        return {
            'total': total,
            'page': page,
            'size': size,
            'pages': (total + size - 1) // size,
            'items': items
        }


def generate_downloads_csv(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        email: Optional[str] = None,
        organisation: Optional[str] = None,
        download_type: Optional[str] = None,
        base_url: Optional[str] = None,
) -> StreamingResponse:
    with Session() as session:
        query = _apply_filters(session.query(DownloadAudit), start_date, end_date, None, email, organisation,
                               download_type)
        downloads = query.order_by(desc(DownloadAudit.timestamp)).all()

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Timestamp', 'Name', 'Email', 'Organisation', 'Type', 'Size', 'Success', 'URL'])

        for d in downloads:
            meta = d.meta or {}
            dtype = meta.get('download_type')
            view_link = ""
            if dtype == 'single_record' and meta.get('doi'):
                view_link = f"{base_url}/{meta.get('doi')}"
            elif dtype == 'zip_bundle' and meta.get('record_ids'):
                query_string = '&'.join([f'record_id_or_doi_list={rid}' for rid in meta.get('record_ids', [])])
                view_link = f"{base_url}/subset?{query_string}"

            writer.writerow([
                d.id, d.timestamp.isoformat(), meta.get('name', ''), meta.get('email', ''),
                meta.get('organisation', ''), dtype, d.file_size or 0, 'Yes' if d.success else 'No', view_link
            ])

        output.seek(0)
        filename = f"download_logs_{datetime.now().date()}.csv"
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )
