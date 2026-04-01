import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from jschon import JSON, URI
from sqlalchemy import select
from starlette.status import HTTP_404_NOT_FOUND

from odp.api.lib.auth import Authorize, Authorized
from odp.api.lib.nextcloud import upload_file_to_nextcloud, delete_folder_from_nextcloud
from odp.api.lib.paging import Paginator
from odp.api.lib.record import create_record
from odp.api.lib.schema import get_metadata_schema
from odp.api.lib.utils import remove_empty_elements
from odp.api.models import (
    RecordModelIn,
)
from odp.api.models import SubmissionModelIn, SubmissionListItemModel
from odp.const import DOI_REGEX
from odp.const import ODPScope, ODPMetadataSchema
from odp.const.db import SubmissionStatus, SchemaType
from odp.db import Session
from odp.db.models import Submission, Schema
from odp.lib.schema import schema_catalog

router = APIRouter()


def submission_list_item_model(submission: Submission) -> SubmissionListItemModel:
    return SubmissionListItemModel(
        id=submission.id,
        title=submission.data.get('title'),
        status=submission.status
    )


# user: list, detail, create, data upload, submit, delete
@router.get(
    '/user_submissions',
    dependencies=[Depends(Authorize(ODPScope.SUBMISSION_READ))],
)
async def list_user_submissions(
        user_id: str,
        paginator: Paginator = Depends(),
):
    stmt = (select(Submission).where(Submission.user_id == user_id))

    return paginator.paginate(
        stmt,
        lambda row: submission_list_item_model(row.Submission),
        sort='submission.id'
    )


@router.get(
    '/{submission_id}',
    dependencies=[Depends(Authorize(ODPScope.SUBMISSION_READ))],
)
async def get_submission(
        submission_id: int,
        user_id: str,
):
    stmt = select(Submission).where(
        Submission.id == submission_id,
        Submission.user_id == user_id
    )

    result = Session.execute(stmt)
    submission = result.scalar_one_or_none()

    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    return submission


@router.post(
    '/',
    dependencies=[Depends(Authorize(ODPScope.SUBMISSION_WRITE))],
)
async def create_submission(
        submission_in: SubmissionModelIn,
):
    submission = Submission(
        data=submission_in.data,
        user_id=submission_in.user_id,
        status=SubmissionStatus.in_progress
    )

    submission.save()

    return submission


@router.put(
    '/{submission_id}',
    dependencies=[Depends(Authorize(ODPScope.SUBMISSION_WRITE))],
)
async def update_submission(
        submission_id: int,
        submission_data: dict[str, Any],
        user_id: str,
):
    statement = select(Submission).where(
        Submission.id == submission_id,
        Submission.user_id == user_id
    )

    result = Session.execute(statement)
    submission = result.scalar_one_or_none()

    if not submission:
        raise HTTPException(HTTP_404_NOT_FOUND)

    submission.data = submission_data

    submission.save()

    return submission


@router.put(
    '/{submission_id}/upload',
    dependencies=[Depends(Authorize(ODPScope.SUBMISSION_WRITE))],
)
async def dataset_upload(
        submission_id: int,
        file: UploadFile = File(...)
):
    if not (submission := Session.get(Submission, submission_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    temp_dir = Path(f'temp_uploads/{submission_id}')

    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)

    local_path_to_file = temp_dir / file.filename

    try:
        with open(local_path_to_file, "wb") as f:
            f.write(file.file.read())

        upload_file_to_nextcloud(local_path_to_file, submission_id, file.filename)

        submission.dataset_file_name = file.filename
        submission.save()
    finally:
        if os.path.exists(local_path_to_file):
            os.remove(local_path_to_file)

        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

        await file.close()


@router.post(
    '/submit/{submission_id}',
    dependencies=[Depends(Authorize(ODPScope.SUBMISSION_WRITE))],
)
async def submit_submission(
        submission_id: int,
        user_id: str,
):
    statement = select(Submission).where(
        Submission.id == submission_id,
        Submission.user_id == user_id
    )

    result = Session.execute(statement)
    submission = result.scalar_one_or_none()

    if not submission:
        raise HTTPException(HTTP_404_NOT_FOUND)

    submission.status = SubmissionStatus.submitted

    submission.save()

    return submission


@router.delete(
    '/{submission_id}',
    dependencies=[Depends(Authorize(ODPScope.SUBMISSION_DELETE))],
)
async def delete_submission(
        submission_id: int,
        user_id: str,
):
    stmt = select(Submission).where(
        Submission.id == submission_id,
        Submission.user_id == user_id
    )

    result = Session.execute(stmt)
    submission = result.scalar_one_or_none()

    if not submission:
        raise HTTPException(HTTP_404_NOT_FOUND)

    delete_folder_from_nextcloud(submission_id)

    submission.delete()


@router.get(
    '/',
    dependencies=[Depends(Authorize(ODPScope.SUBMISSION_ADMIN))],
)
async def list_submissions(
        paginator: Paginator = Depends(),
        status: str = 'all'
):
    stmt = (select(Submission))

    if not status == 'all' and status in [s.value for s in SubmissionStatus]:
        stmt = stmt.where(Submission.status == status)

    return paginator.paginate(
        stmt,
        lambda row: submission_list_item_model(row.Submission),
        sort='submission.id'
    )


@router.get(
    '/admin/{submission_id}',
    dependencies=[Depends(Authorize(ODPScope.SUBMISSION_ADMIN))],
)
async def admin_get_submission(
        submission_id: int
):
    if not (submission := Session.get(Submission, submission_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    return submission


@router.put(
    '/admin/{submission_id}',
    dependencies=[Depends(Authorize(ODPScope.SUBMISSION_ADMIN))],
)
async def admin_update_submission(
        submission_id: int,
        submission_in: SubmissionModelIn,
):
    if not (submission := Session.get(Submission, submission_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    submission.data = submission_in.data
    submission.status = submission_in.status if submission_in.status else submission.status
    submission.collection_id = submission_in.collection_id if submission_in.collection_id else submission.collection_id
    submission.schema_id = submission_in.schema_id if submission_in.schema_id else submission.schema_id

    submission.save()

    return submission


@router.delete(
    '/admin/{submission_id}',
    dependencies=[Depends(Authorize(ODPScope.SUBMISSION_ADMIN))],
)
async def admin_delete_submission(
        submission_id: int,
):
    if not (submission := Session.get(Submission, submission_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    delete_folder_from_nextcloud(submission_id)

    submission.delete()


@router.put(
    '/admin/{submission_id}/accept',
    dependencies=[Depends(Authorize(ODPScope.SUBMISSION_ADMIN))],
)
async def accept_submission(
        submission_id: int,
        collection_id: str,
        schema_id: ODPMetadataSchema,
        auth: Authorized = Depends(Authorize(ODPScope.RECORD_READ)),
        doi: str = Query(..., pattern=DOI_REGEX)
) -> bool:
    if not (submission := Session.get(Submission, submission_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    submission.collection_id = collection_id
    submission.schema_id = schema_id
    submission.doi = doi
    submission.save()

    _add_additional_metadata_fields(submission, schema_id == ODPMetadataSchema.SAEON_ISO19115)

    cleaned_metadata = remove_empty_elements(submission.data)

    schema = Session.get(Schema, (ODPMetadataSchema.SAEON_DATA_SUBMISSION, SchemaType.metadata))
    data_submission_schema = schema_catalog.get_schema(URI(schema.uri))
    result = data_submission_schema.evaluate(JSON(cleaned_metadata))
    translated_metadata = result.output('translation', scheme='saeon/datacite4', ignore_validity=True)

    record_in = RecordModelIn(
        doi=doi,
        collection_id=collection_id,
        schema_id=ODPMetadataSchema.SAEON_DATACITE4,
        metadata=translated_metadata
    )

    datacite_schema = await get_metadata_schema(record_in)

    created_record = create_record(record_in, datacite_schema, auth)

    submission.record_id = created_record.id
    submission.status = SubmissionStatus.accepted
    submission.save()

    return True


def _add_additional_metadata_fields(submission: Submission, is_iso: bool = False):
    submission.data['timestamp'] = datetime.now().strftime("%Y-%m-%d")
    submission.data['language'] = 'en-us'
    submission.data['doi'] = submission.doi

    if is_iso:
        if 'related_identifiers' not in submission.data:
            submission.data['related_identifiers'] = []
        submission.data['related_identifiers'].append({
            "related_identifier": "https://odp.saeon.ac.za/schema/metadata/saeon/iso19115",
            "relationship_type": "HasMetadata",
            "related_metadata_scheme": "ISO 19115-1",
            "scheme_uri": "https://schemas.isotc211.org/19115/",
            "scheme_type": "JSON"
        })
