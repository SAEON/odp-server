from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.status import HTTP_201_CREATED, HTTP_400_BAD_REQUEST
from sqlalchemy import insert

from odp.db import Session
from odp.db.models import DownloadAudit
from odp.api.lib.auth import Authorize, Authorized  # only if you want auth; otherwise omit

router = APIRouter()

@router.post('/audit', status_code=HTTP_201_CREATED)
async def create_download_audit(request: Request):
    """
    Accept JSON payload to record a download audit and return success.
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
    for k in ('name', 'email', 'organisation'):
        if payload.get(k) is not None:
            meta[k] = payload.get(k)

    # capture IP and UA
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get('user-agent')

    # persist using Session
    #
    # --- THIS IS THE FIX ---
    # We use `Session.begin()` to manage the transaction,
    # and call methods on the imported `Session` object itself.
    #
    with Session.begin():
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
        Session.add(audit)
        Session.flush()
        audit_id = audit.id
    # --- END OF FIX ---

    return {"status": "ok", "audit_id": audit_id}