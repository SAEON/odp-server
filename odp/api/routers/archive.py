import hashlib
import pathlib
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from starlette.status import HTTP_404_NOT_FOUND, HTTP_405_METHOD_NOT_ALLOWED, HTTP_422_UNPROCESSABLE_ENTITY

from odp.api.lib.archive import ArchiveAdapter, get_archive_adapter
from odp.api.lib.auth import Authorize
from odp.api.lib.paging import Paginator
from odp.api.models import ArchiveModel, ArchiveResourceModel, Page, ResourceModel
from odp.api.routers.resource import output_resource_model
from odp.const import ODPScope
from odp.db import Session
from odp.db.models import Archive, ArchiveResource, Package, PackageResource, Resource

router = APIRouter()


def output_archive_model(result) -> ArchiveModel:
    return ArchiveModel(
        id=result.Archive.id,
        url=result.Archive.url,
        resource_count=result.count,
    )


def output_archive_resource_model(result) -> ArchiveResourceModel:
    return ArchiveResourceModel(
        archive_id=result.ArchiveResource.archive_id,
        resource_id=result.ArchiveResource.resource_id,
        path=result.ArchiveResource.path,
        title=result.Resource.title,
        description=result.Resource.description,
        filename=result.Resource.filename,
        mimetype=result.Resource.mimetype,
        size=result.Resource.size,
        md5=result.Resource.md5,
        timestamp=result.Resource.timestamp.isoformat(),
        provider_id=result.Resource.provider_id,
        provider_key=result.Resource.provider.key,
    )


@router.get(
    '/',
    response_model=Page[ArchiveModel],
    dependencies=[Depends(Authorize(ODPScope.ARCHIVE_READ))],
)
async def list_archives(
        paginator: Paginator = Depends(),
):
    stmt = (
        select(Archive, func.count(ArchiveResource.archive_id)).
        outerjoin(ArchiveResource).
        group_by(Archive)
    )

    return paginator.paginate(
        stmt,
        lambda row: output_archive_model(row),
        sort_model=Archive,
    )


@router.get(
    '/{archive_id}',
    response_model=ArchiveModel,
    dependencies=[Depends(Authorize(ODPScope.ARCHIVE_READ))],
)
async def get_archive(
        archive_id: str,
):
    stmt = (
        select(Archive, func.count(ArchiveResource.archive_id)).
        outerjoin(ArchiveResource).
        group_by(Archive).
        where(Archive.id == archive_id)
    )

    if not (result := Session.execute(stmt).one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_archive_model(result)


@router.get(
    '/{archive_id}/resources',
    response_model=Page[ArchiveResourceModel],
    dependencies=[Depends(Authorize(ODPScope.ARCHIVE_READ))],
)
async def list_resources(
        archive_id: str,
        paginator: Paginator = Depends(),
):
    if not Session.get(Archive, archive_id):
        raise HTTPException(HTTP_404_NOT_FOUND)

    stmt = (
        select(ArchiveResource, Resource).join(Resource).
        where(ArchiveResource.archive_id == archive_id)
    )

    return paginator.paginate(
        stmt,
        lambda row: output_archive_resource_model(row),
    )


@router.put(
    '/{archive_id}/{package_id}/{path:path}',
)
async def add_resource(
        archive_id: str,
        package_id: str,
        path: str = Path(..., title='Archival path, relative to the archive base URL'),
        title: str = Query(..., title='Resource title'),
        description: str = Query(None, title='Resource description'),
        filename: str = Query(None, title='File name'),
        mimetype: str = Query(None, title='Content type'),
        size: int = Query(None, title='File size'),
        md5: str = Query(None, title='MD5 checksum'),
        file: UploadFile = File(None, title='File upload'),
        archive_adapter: ArchiveAdapter = Depends(get_archive_adapter),
) -> ResourceModel:
    """
    Add a resource to a package and register it as archived.
    Resource metadata may be provided via query parameters.

    If the request includes a file upload, it is stored to the
    archive; upload size and computed checksum are verified
    against the size and md5 parameters, if supplied.

    The relative path to the resource within the package is set
    to the filename, if supplied; otherwise, the archival path
    is used.
    """
    # todo: archive-specific scopes
    if not (archive := Session.get(Archive, archive_id)):
        raise HTTPException(
            HTTP_404_NOT_FOUND, 'Archive not found'
        )

    # todo: check provider auth
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(
            HTTP_404_NOT_FOUND, 'Package not found'
        )

    if not path:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY, "path cannot be blank"
        )

    if '..' in path:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY, "'..' not allowed in path"
        )

    if pathlib.Path(path).is_absolute():
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY, 'path must be relative'
        )

    if file is not None:
        if size is not None and size != file.size:
            raise HTTPException(
                HTTP_422_UNPROCESSABLE_ENTITY,
                f'Upload file size ({file.size}) does not match the given size ({size})'
            )
        size = file.size

        file_md5 = hashlib.md5(await file.read()).hexdigest()
        if md5 is not None and md5 != file_md5:
            raise HTTPException(
                HTTP_422_UNPROCESSABLE_ENTITY,
                f"Upload file MD5 checksum '{file_md5}' does not match the given MD5 checksum '{md5}'"
            )
        md5 = file_md5

    resource = Resource(
        title=title,
        description=description,
        filename=filename,
        mimetype=mimetype,
        size=size,
        md5=md5,
        timestamp=(timestamp := datetime.now(timezone.utc)),
        provider_id=package.provider_id,
    )
    resource.save()

    try:
        archive_resource = ArchiveResource(
            archive_id=archive_id,
            resource_id=resource.id,
            path=path,
            timestamp=timestamp,
        )
        archive_resource.save()
    except IntegrityError:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY, f'Path {path} already exists in archive'
        )

    try:
        package_resource = PackageResource(
            package_id=package_id,
            resource_id=resource.id,
            path=(package_path := (filename or path)),
            timestamp=timestamp,
        )
        package_resource.save()
    except IntegrityError:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY, f'Path {package_path} already exists in package'
        )

    if file is not None:
        try:
            await archive_adapter.put(path, file, md5)
        except NotImplementedError:
            raise HTTPException(
                HTTP_405_METHOD_NOT_ALLOWED, f'Operation not supported for {archive_id}'
            )

    return output_resource_model(resource)
