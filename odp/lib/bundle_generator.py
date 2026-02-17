"""
Server-side ZIP bundle generation with metadata PDFs and data files.
"""

import logging
import re
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile, ZIP_DEFLATED

import requests
from fastapi import HTTPException
from sqlalchemy import select

from odp.db import Session
from odp.db.models import CatalogRecord, DownloadAudit
from odp.lib.metadata_adapters import adapt_metadata
from odp.lib.pdf_generator import generate_pdf

logger = logging.getLogger(__name__)


def create_safe_folder_name(title: str, max_length: int = 200) -> str:
    """Sanitize title for ZIP folder structure."""
    if not title or not isinstance(title, str):
        return 'Untitled'

    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '', title.strip())
    sanitized = sanitized.replace(' ', '_')
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')

    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')

    return sanitized or 'Untitled'


def fetch_external_file(url: str, timeout: int = 30) -> Optional[bytes]:
    """Download a file with logging for HTTP failures."""
    try:
        response = requests.get(url + '/download', timeout=timeout, stream=True)
        if not response.ok:
            logger.warning(f"Download failed (HTTP {response.status_code}): {url}")
            return None

        file_bytes = b''
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_bytes += chunk
        return file_bytes
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return None


def create_zip_bundle(
        record_ids: List[str],
        user_data: Dict[str, str],
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Generates ZIP bundle with metadata PDFs and data files using bulk database fetching."""

    zip_buffer = BytesIO()
    total_file_size = 0
    processed_dois = []
    failed_records = []
    MAX_BUNDLE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

    with Session() as session:
        # Step 1: Bulk Fetch all published records in one DB call
        # Re think and use Record
        stmt = (
            select(CatalogRecord)
            .where(CatalogRecord.published == True)
            .where(CatalogRecord.record_id.in_(record_ids))
            .where(CatalogRecord.catalog_id == 'DataCite')
        )
        catalog_records = session.execute(stmt).scalars().all()

        # Track which IDs were actually found in the DB
        found_ids = {rec.record_id for rec in catalog_records}
        for missing_id in set(record_ids) - found_ids:
            failed_records.append({'doi': missing_id, 'reason': 'not_found_or_not_published'})

        with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zip_file:
            for catalog_record in catalog_records:
                try:
                    pub_rec = catalog_record.published_record

                    # Discover metadata: prefer metadata_records, fallback to root
                    metadata = None
                    if pub_rec.get("metadata_records"):
                        metadata = pub_rec["metadata_records"][0].get("metadata")
                    elif pub_rec.get("metadata"):
                        metadata = pub_rec.get("metadata")

                    if not metadata:
                        logger.error(f"No metadata found for {catalog_record.record_id}")
                        failed_records.append({'doi': catalog_record.record_id, 'reason': 'metadata_missing'})
                        continue

                    doi = catalog_record.record.doi or catalog_record.record_id
                    resource = pub_rec.get("immutableResource")

                    # Step 2: Title extraction and folder sanitization
                    raw_title = (
                            metadata.get('titles', [{}])[0].get('title') or
                            metadata.get('title') or
                            f"Record_{doi}"
                    )
                    folder_name = create_safe_folder_name(raw_title)

                    # Step 3: PDF Generation
                    try:
                        # Use adapted generator/adapter pattern
                        record_metadata = adapt_metadata(metadata, schema_id='auto')
                        pdf_buffer = generate_pdf(record_metadata)
                        pdf_content = pdf_buffer.getvalue()

                        zip_file.writestr(f"{folder_name}/metadata.pdf", pdf_content)
                        total_file_size += len(pdf_content)
                    except Exception as e:
                        logger.error(f"PDF generation failed for {catalog_record.record_id}: {str(e)}")

                    # Step 4: Resource Download
                    if resource and 'resourceDownload' in resource:
                        download_url = resource['resourceDownload'].get('downloadURL')
                        file_name = resource['resourceDownload'].get('fileName', 'data_file')
                        if download_url:
                            file_bytes = fetch_external_file(download_url)
                            if file_bytes:
                                zip_file.writestr(f"{folder_name}/{file_name}", file_bytes)
                                total_file_size += len(file_bytes)

                    processed_dois.append(doi)

                    if total_file_size > MAX_BUNDLE_SIZE:
                        raise HTTPException(413, 'Bundle exceeds 2GB limit')

                except Exception as e:
                    logger.error(f"Error processing {catalog_record.record_id}: {str(e)}")
                    failed_records.append({'doi': catalog_record.record_id, 'reason': 'internal_error'})

    final_zip = zip_buffer.getvalue()

    # Step 5: Simplified Audit Logging
    try:
        log_bundle_download_audit(
            record_ids,
            processed_dois,
            user_data,
            len(final_zip),
            failed_records,
            client_ip,
            user_agent
        )
    except Exception as e:
        logger.error(f"Failed to log audit: {str(e)}")

    return final_zip, {
        'total_size': len(final_zip),
        'record_count': len(processed_dois),
        'failed_count': len(failed_records),
        'processed': processed_dois,
        'failed': failed_records
    }


def log_bundle_download_audit(record_ids, dois, user_data, file_size, failed_records, client_ip, user_agent):
    """Logs the ZIP generation event."""

    # Determine if this is a single record download or a bulk bundle
    # We check 'dois' (successful lookups) to ensure we have valid data
    is_single_record = len(record_ids) == 1

    with Session() as session:
        audit_meta = {
            'name': user_data.get('name'),
            'email': user_data.get('email'),
            'organisation': user_data.get('organisation'),
            'failed_records': failed_records,
        }

        # Switch logic based on count
        if is_single_record:
            audit_meta['download_type'] = 'single_record'
            audit_meta['doi'] = dois[0]
            # Store the input identifier as well, just in case
            if len(record_ids) > 0:
                audit_meta['record_id'] = record_ids[0]
        else:
            audit_meta['download_type'] = 'zip_bundle'
            audit_meta['record_ids'] = record_ids
            audit_meta['dois'] = dois

        audit = DownloadAudit(
            client_id='odp-server-zip-generator',
            download_url='/catalog/generate-zip-bundle',
            ip_address=client_ip,
            user_agent=user_agent,
            file_size=file_size,
            success=True,
            timestamp=datetime.now(timezone.utc),
            meta=audit_meta
        )
        session.add(audit)
        session.commit()
