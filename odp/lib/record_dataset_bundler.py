"""
Metadata bundle generation for client-side ZIP downloads.
"""

import base64
import logging
import mimetypes
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from sqlalchemy import select

from odp.api.models import MetadataBundleRecord, MetadataBundleResponse
from odp.const import ODPCatalog
from odp.db import Session
from odp.db.models import CatalogRecord, DownloadAudit
from odp.lib.metadata_adapters import adapt_metadata
from odp.lib.pdf_generator import generate_pdf

logger = logging.getLogger(__name__)


def _sanitize_name_part(value: str) -> str:
    """Sanitize a single name component for use in a ZIP folder/file path."""
    if not value or not isinstance(value, str):
        return ''

    value = value.strip().replace('/', '_').replace('\\', '_')
    sanitized = re.sub(r'[<>:"|?*]', '', value)
    sanitized = sanitized.replace(' ', '_')
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_')


def create_safe_folder_name(doi: str, title: str, max_length: int = 200) -> str:
    """Build a sanitized '{doi}_{title}' folder name for the ZIP structure.

    The DOI is always preserved in full, with only the title truncated if
    needed, so that a truncated folder name still uniquely identifies the
    record.
    """
    safe_doi = _sanitize_name_part(doi)
    safe_title = _sanitize_name_part(title)

    if safe_doi and safe_title:
        name = f'{safe_doi}_{safe_title}'
        if len(name) > max_length:
            remaining = max_length - len(safe_doi) - 1
            name = f'{safe_doi}_{safe_title[:remaining].rstrip("_")}' if remaining > 0 else safe_doi[:max_length]
    else:
        name = safe_doi or safe_title
        if len(name) > max_length:
            name = name[:max_length].rstrip('_')

    return name or 'Untitled'


def _ensure_extension(file_name: str, download_url: str, content_type: str) -> str:
    """Append a file extension if file_name has none, using URL path or Content-Type."""
    if os.path.splitext(file_name)[1]:
        return file_name

    # Try extension from URL path
    url_path = urlparse(download_url).path
    ext = os.path.splitext(url_path)[1]

    # Fall back to Content-Type
    if not ext and content_type:
        ext = mimetypes.guess_extension(content_type) or ''
        # mimetypes can return platform-specific oddities; normalise common ones
        ext = {'.jpe': '.jpg', '.jpeg': '.jpg'}.get(ext, ext)

    return file_name + ext if ext else file_name


def _extract_metadata(pub_rec: dict) -> dict | None:
    if pub_rec.get("metadata_records"):
        return pub_rec["metadata_records"][0].get("metadata")
    return pub_rec.get("metadata")


def log_bundle_download_audit(record_ids, dois, user_data, file_size, failed_records, client_ip, user_agent, catalog_url=None):
    """Logs the ZIP generation event."""
    is_single = len(record_ids) == 1

    download_type = 'single_record' if is_single else 'zip_bundle'
    doi_data = {'doi': dois[0], 'record_id': record_ids[0]} if is_single else {'record_ids': record_ids, 'dois': dois}

    audit_meta = {
        'name': user_data.get('name'),
        'email': user_data.get('email'),
        'organisation': user_data.get('organisation'),
        'failed_records': failed_records,
        'download_type': download_type,
        **doi_data,
    }
    if catalog_url:
        audit_meta['catalog_url'] = catalog_url

    DownloadAudit(
        client_id='odp-server-zip-generator',
        download_url='/catalog/metadata-bundle',
        ip_address=client_ip,
        user_agent=user_agent,
        file_size=file_size,
        success=True,
        timestamp=datetime.now(timezone.utc),
        meta=audit_meta,
    ).save()

    # Commit here rather than relying on the request-scoped commit middleware:
    # this function runs in a threadpool worker thread (metadata_bundle is
    # a sync endpoint), so the thread-local Session it uses is not the same
    # one the middleware commits on the event-loop thread.
    try:
        Session.commit()
    except Exception:
        Session.rollback()
        raise


def generate_metadata_bundle(
        record_ids: list[str],
        user_data: dict[str, str],
        client_ip: str | None = None,
        user_agent: str | None = None,
        catalog_url: str | None = None,
) -> MetadataBundleResponse:
    """Return metadata PDFs (base64) + data file URLs per record. Does NOT download data files."""
    if not record_ids:
        raise ValueError("record_ids cannot be empty")
    required = {'name', 'email', 'organisation'}
    if not required.issubset(user_data) or not all(user_data.get(f) for f in required):
        raise ValueError("user_data missing required fields: name, email, organisation")

    records = []
    failed_records = []
    processed_dois = []

    with Session() as session:
        stmt = (
            select(CatalogRecord)
            .where(CatalogRecord.published.is_(True))
            .where(CatalogRecord.record_id.in_(record_ids))
            .where(CatalogRecord.catalog_id == ODPCatalog.DATACITE)
        )
        catalog_records = session.execute(stmt).scalars().all()

        found_ids = {rec.record_id for rec in catalog_records}
        for missing_id in set(record_ids) - found_ids:
            failed_records.append({'doi': missing_id, 'reason': 'not_found_or_not_published'})

        for catalog_record in catalog_records:
            try:
                pub_rec = catalog_record.published_record
                metadata = _extract_metadata(pub_rec)

                if not metadata:
                    failed_records.append({'doi': catalog_record.record_id, 'reason': 'metadata_missing'})
                    continue

                doi = catalog_record.record.doi or catalog_record.record_id
                raw_title = (
                    metadata.get('titles', [{}])[0].get('title') or
                    metadata.get('title') or
                    f"Record_{doi}"
                )
                folder_name = create_safe_folder_name(doi, raw_title)

                record_metadata = adapt_metadata(metadata)
                pdf_buffer = generate_pdf(record_metadata)
                pdf_b64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')

                resource = metadata.get('immutableResource')
                data_file_url = None
                data_file_name = None
                if resource and 'resourceDownload' in resource:
                    data_file_url = resource['resourceDownload'].get('downloadURL')
                    data_file_name = resource['resourceDownload'].get('fileName', 'data_file')
                    if data_file_url and data_file_name:
                        data_file_name = _ensure_extension(data_file_name, data_file_url, None)

                records.append(MetadataBundleRecord(
                    folder_name=folder_name,
                    metadata_pdf=pdf_b64,
                    data_file_url=data_file_url,
                    data_file_name=data_file_name,
                ))
                processed_dois.append(doi)

            except Exception as e:
                logger.error(f"Error processing {catalog_record.record_id}: {e}")
                failed_records.append({'doi': catalog_record.record_id, 'reason': 'internal_error'})

    log_bundle_download_audit(
        record_ids=record_ids,
        dois=processed_dois,
        user_data=user_data,
        file_size=0,
        failed_records=failed_records,
        client_ip=client_ip,
        user_agent=user_agent,
        catalog_url=catalog_url,
    )

    return MetadataBundleResponse(
        records=records,
        total=len(records),
        failed=len(failed_records),
    )
