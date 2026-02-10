"""
Server-side ZIP bundle generation with metadata PDFs and data files.
"""

import logging
import re
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape

import requests
from fastapi import HTTPException
from sqlalchemy import select, func

from odp.db import Session
from odp.db.models import CatalogRecord, DownloadAudit, Record
from odp.lib.metadata_adapters import adapt_metadata
from odp.lib.metadata_pdf import generate_pdf

logger = logging.getLogger(__name__)


def fetch_record_data(id_or_doi: str) -> Optional[Dict[str, Any]]:
    try:
        with Session() as session:
            stmt = select(CatalogRecord).where(CatalogRecord.published == True)

            try:
                UUID(id_or_doi, version=4)
                stmt = stmt.where(CatalogRecord.record_id == id_or_doi)
            except ValueError:
                stmt = stmt.join(Record).where(func.lower(Record.doi) == id_or_doi.lower())

            catalog_record = session.execute(stmt).scalars().first()

            if not catalog_record or not catalog_record.published_record:
                logger.error(f"Record not found or not published: {id_or_doi}")
                return None

            pub_rec = catalog_record.published_record

            # Discover metadata: prefer metadata_records (DataCite), fallback to root metadata
            metadata = None

            if pub_rec.get("metadata_records") and len(pub_rec["metadata_records"]) > 0:
                metadata = pub_rec["metadata_records"][0].get("metadata")
            elif pub_rec.get("metadata"):
                metadata = pub_rec.get("metadata")

            if not metadata:
                available_keys = list(pub_rec.keys())
                logger.error(f"No recognizable metadata for {id_or_doi}. Keys found: {available_keys}")
                return None

            return {
                "metadata": metadata,
                "resource": pub_rec.get("immutableResource"),
                "doi": catalog_record.record.doi or catalog_record.record_id,
                "catalog_id": catalog_record.catalog_id
            }
    except Exception as e:
        logger.error(f"Error fetching record {id_or_doi}: {str(e)}")
        return None
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
        response = requests.get(url+'/download', timeout=timeout, stream=True)
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
    """Generates ZIP bundle with metadata PDFs and data files."""

    zip_buffer = BytesIO()
    total_file_size = 0
    processed_records = []
    failed_records = []
    MAX_BUNDLE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

    with ZipFile(zip_buffer, 'w', ZIP_DEFLATED) as zip_file:
        for record_id in record_ids:
            try:
                # 1. Fetch
                data_package = fetch_record_data(record_id)
                if not data_package:
                    failed_records.append({'doi': record_id, 'reason': 'not_found_or_not_published'})
                    continue

                metadata = data_package['metadata']
                resource = data_package['resource']

                # 2. Title extraction for folder
                raw_title = (
                    metadata.get('titles', [{}])[0].get('title') or
                    metadata.get('title') or
                    f"Record_{data_package['doi']}"
                )
                folder_name = create_safe_folder_name(raw_title)

                # 3. PDF Assembly
                try:
                    safe_metadata = metadata.copy()
                    if 'titles' in safe_metadata and safe_metadata['titles']:
                        safe_metadata['titles'][0]['title'] = escape(raw_title)
                    elif 'title' in safe_metadata:
                        safe_metadata['title'] = escape(raw_title)

                    record_metadata = adapt_metadata(safe_metadata, schema_id='auto')
                    pdf_buffer = generate_pdf(record_metadata)
                    pdf_content = pdf_buffer.getvalue()

                    zip_file.writestr(f"{folder_name}/metadata.pdf", pdf_content)
                    total_file_size += len(pdf_content)
                except Exception as e:
                    logger.error(f"PDF generation failed for {record_id}: {str(e)}")

                # 4. Resource Download
                if resource and 'resourceDownload' in resource:
                    download_url = resource['resourceDownload'].get('downloadURL')
                    file_name = resource['resourceDownload'].get('fileName', 'data_file')
                    if download_url:
                        file_bytes = fetch_external_file(download_url)
                        if file_bytes:
                            zip_file.writestr(f"{folder_name}/{file_name}", file_bytes)
                            total_file_size += len(file_bytes)
                        else:
                            logger.warning(f"Data file download returned no bytes for {record_id}")
                    else:
                        logger.warning(f"Resource download URL missing for {record_id}")

                processed_records.append(data_package['doi'])

                if total_file_size > MAX_BUNDLE_SIZE:
                    logger.error(f"Bundle size exceeds 2GB limit ({total_file_size} bytes).")
                    raise HTTPException(413, 'Bundle exceeds 2GB limit')

            except Exception as e:
                logger.error(f"Error processing record {record_id}: {str(e)}")
                failed_records.append({'doi': record_id, 'reason': 'internal_error'})

    final_zip = zip_buffer.getvalue()

    try:
        log_bundle_download_audit(record_ids,processed_records, user_data, len(final_zip), failed_records, client_ip, user_agent)
    except Exception as e:
        logger.error(f"Failed to log audit for bundle download: {str(e)}")

    return final_zip, {
        'total_size': len(final_zip),
        'record_count': len(processed_records),
        'failed_count': len(failed_records),
        'processed': processed_records,
        'failed': failed_records
    }


def log_bundle_download_audit(record_ids, dois, user_data, file_size, failed_records, client_ip, user_agent):
    """Logs the ZIP generation event."""

    # Determine if this is a single record download or a bulk bundle
    # We check 'dois' (successful lookups) to ensure we have valid data
    is_single_record = len(dois) == 1

    with Session() as session:
        meta_data = {
            'name': user_data.get('name'),
            'email': user_data.get('email'),
            'organisation': user_data.get('organisation'),
            'failed_records': failed_records,
        }

        # Switch logic based on count
        if is_single_record:
            meta_data['download_type'] = 'single_record'
            meta_data['doi'] = dois[0]
            # Store the input identifier as well, just in case
            if len(record_ids) > 0:
                meta_data['record_id'] = record_ids[0]
        else:
            meta_data['download_type'] = 'zip_bundle'
            meta_data['record_ids'] = record_ids
            meta_data['dois'] = dois

        audit = DownloadAudit(
            client_id='odp-server-zip-generator',
            download_url='/catalog/generate-zip-bundle',
            ip_address=client_ip,
            user_agent=user_agent,
            file_size=file_size,
            success=True,
            timestamp=datetime.now(timezone.utc),
            meta=meta_data
        )
        session.add(audit)
        session.commit()