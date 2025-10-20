import mimetypes
import pathlib
from datetime import datetime, timezone
from typing import BinaryIO

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile
from jschon import JSON, JSONPatch, URI
from odp.api.lib.remove_empty_children import remove_empty_children
#from jschon_translation import remove_empty_children
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from starlette.responses import StreamingResponse
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_405_METHOD_NOT_ALLOWED, HTTP_422_UNPROCESSABLE_ENTITY
from werkzeug.utils import secure_filename

from odp.api.lib.auth import ArchiveAuthorize, Authorize, Authorized, TagAuthorize, UntagAuthorize
from odp.api.lib.paging import Paginator
from odp.api.lib.schema import get_metadata_validity
from odp.api.lib.tagging import Tagger, output_tag_instance_model
from odp.api.models import PackageDetailModel, PackageModel, PackageModelIn, Page, TagInstanceModel, TagInstanceModelIn
from odp.api.routers.resource import output_resource_model
from odp.const import ODPScope
from odp.const.db import ArchiveResourceStatus, HashAlgorithm, PackageCommand, PackageStatus, ResourceStatus, SchemaType, TagType
from odp.db import Session
from odp.db.models import Archive, ArchiveResource, Package, PackageAudit, Provider, Resource, Schema
from odp.lib.archive import ArchiveAdapter, ArchiveError, ArchiveFileResponse
from odp.lib.schema import schema_catalog

router = APIRouter()


def output_package_model(package: Package, *, detail=False) -> PackageModel | PackageDetailModel:
    cls = PackageDetailModel if detail else PackageModel
    record = next((r for r in package.records), None)
    kwargs = dict(
        id=package.id,
        key=package.key,
        status=package.status,
        timestamp=package.timestamp.isoformat(),
        provider_id=package.provider_id,
        provider_key=package.provider.key,
        resource_ids=[resource.id for resource in package.resources],
        schema_id=package.schema_id,
        schema_uri=package.schema.uri,
        record_id=record.id if record else None,
        record_doi=record.doi if record else None,
        record_sid=record.sid if record else None,
        resources=[
            output_resource_model(resource)
            for resource in package.resources
        ],
        tags=[
            output_tag_instance_model(package_tag)
            for package_tag in package.tags
        ],
    )
    if detail:
        kwargs |= dict(
            metadata=package.metadata_,
            validity=package.validity,
        )

    return cls(**kwargs)


def create_audit_record(
        auth: Authorized,
        package: Package,
        timestamp: datetime,
        command: PackageCommand,
) -> None:
    PackageAudit(
        client_id=auth.client_id,
        user_id=auth.user_id,
        command=command,
        timestamp=timestamp,
        _id=package.id,
        _key=package.key,
        _status=package.status,
        _provider_id=package.provider_id,
        _schema_id=package.schema_id,
        _resources=[resource.path for resource in package.resources],
    ).save()


def ensure_status(package: Package, status: PackageStatus) -> None:
    if package.status != status:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY,
            f"Package status must be '{status}'"
        )


@router.get(
    '/',
)
async def list_packages(
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_READ)),
        provider_id: str = None,
        paginator: Paginator = Depends(),
) -> Page[PackageModel]:
    """
    List provider-accessible packages. Requires scope `odp.package:read`.
    """
    stmt = (
        select(Package)
        .where(Package.status != PackageStatus.delete_pending)
    )

    if auth.object_ids != '*':
        stmt = stmt.where(Package.provider_id.in_(auth.object_ids))

    if provider_id:
        auth.enforce_constraint([provider_id])
        stmt = stmt.where(Package.provider_id == provider_id)

    return paginator.paginate(
        stmt,
        lambda row: output_package_model(row.Package),
    )


@router.get(
    '/all/',
    dependencies=[Depends(Authorize(ODPScope.PACKAGE_READ_ALL))],
)
async def list_all_packages(
        provider_id: str = None,
        paginator: Paginator = Depends(),
) -> Page[PackageModel]:
    """
    List all packages. Requires scope `odp.package:read_all`.
    """
    stmt = select(Package)

    if provider_id:
        stmt = stmt.where(Package.provider_id == provider_id)

    return paginator.paginate(
        stmt,
        lambda row: output_package_model(row.Package),
    )


@router.get(
    '/{package_id}',
)
async def get_package(
        package_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_READ)),
) -> PackageDetailModel:
    """
    Get a provider-accessible package. Requires scope `odp.package:read`.
    """
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    auth.enforce_constraint([package.provider_id])

    return output_package_model(package, detail=True)


@router.get(
    '/all/{package_id}',
    dependencies=[Depends(Authorize(ODPScope.PACKAGE_READ_ALL))],
)
async def get_any_package(
        package_id: str,
) -> PackageDetailModel:
    """
    Get any package. Requires scope `odp.package:read_all`.
    """
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_package_model(package, detail=True)


@router.post(
    '/',
)
async def create_package(
        package_in: PackageModelIn,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_WRITE)),
) -> PackageDetailModel:
    """
    Create a provider-accessible package. Requires scope `odp.package:write`.
    """
    return await _create_package(package_in, auth)


@router.post(
    '/admin/',
)
async def admin_create_package(
        package_in: PackageModelIn,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_ADMIN)),
) -> PackageDetailModel:
    """
    Create a package for any provider. Requires scope `odp.package:admin`.
    """
    return await _create_package(package_in, auth)


async def _create_package(
        package_in: PackageModelIn,
        auth: Authorized,
):
    auth.enforce_constraint([package_in.provider_id])

    if not (provider := Session.get(Provider, package_in.provider_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    timestamp = datetime.now(timezone.utc)
    date = timestamp.strftime('%Y_%m_%d')
    n = 1
    while True:
        try:
            package = Package(
                key=f'{provider.key}_{date}_{n:03}',
                status=PackageStatus.editing,
                timestamp=timestamp,
                provider_id=package_in.provider_id,
                schema_id=package_in.schema_id,
                schema_type=SchemaType.metadata,
            )
            package.save()
            break
        except IntegrityError:
            Session.rollback()
            n += 1

    create_audit_record(auth, package, timestamp, PackageCommand.insert)

    return output_package_model(package, detail=True)


@router.put(
    '/admin/{package_id}',
)
async def admin_update_package(
        package_id: str,
        package_in: PackageModelIn,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_ADMIN)),
) -> PackageDetailModel:
    """
    Update any package. Requires scope `odp.package:admin`.
    """
    return await _update_package(package_id, package_in, auth)


async def _update_package(
        package_id: str,
        package_in: PackageModelIn,
        auth: Authorized,
):
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    auth.enforce_constraint([package.provider_id, package_in.provider_id])

    if (
            package.provider_id != package_in.provider_id or
            package.schema_id != package_in.schema_id
    ):
        package.timestamp = (timestamp := datetime.now(timezone.utc))
        package.provider_id = package_in.provider_id
        package.schema_id = package_in.schema_id
        package.save()
        create_audit_record(auth, package, timestamp, PackageCommand.update)

    return output_package_model(package, detail=True)


@router.delete(
    '/{package_id}',
)
async def delete_package(
        package_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_WRITE)),
) -> None:
    """
    Delete a provider-accessible package. The package status must be `editing`.
    Requires scope `odp.package:write`.
    """
    await _delete_package(package_id, auth, True)


@router.delete(
    '/admin/{package_id}',
)
async def admin_delete_package(
        package_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_ADMIN)),
) -> None:
    """
    Delete any package. Requires scope `odp.package:admin`.
    """
    await _delete_package(package_id, auth, False)


async def _delete_package(
        package_id: str,
        auth: Authorized,
        check_status: bool,
):
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    auth.enforce_constraint([package.provider_id])

    if check_status:
        ensure_status(package, PackageStatus.editing)

    if package.package_records:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY,
            'A package with an associated record cannot be deleted.',
        )

    timestamp = datetime.now(timezone.utc)
    create_audit_record(auth, package, timestamp, PackageCommand.delete)

    package.status = PackageStatus.delete_pending
    package.timestamp = timestamp
    package.save()

    for resource in package.resources:
        if resource.status != ResourceStatus.delete_pending:
            resource.status = ResourceStatus.delete_pending
            resource.timestamp = timestamp
            resource.save()


@router.post(
    '/{package_id}/tag',
)
async def tag_package(
        package_id: str,
        tag_instance_in: TagInstanceModelIn,
        auth: Authorized = Depends(TagAuthorize()),
) -> TagInstanceModel | None:
    """
    Set a tag instance on a package, returning the created or updated instance,
    or null if no change was made. The package status must be `editing`. Requires
    the scope associated with the tag.
    """
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    auth.enforce_constraint([package.provider_id])

    ensure_status(package, PackageStatus.editing)

    if package_tag := await Tagger(TagType.package).set_tag_instance(tag_instance_in, package, auth):
        return output_tag_instance_model(package_tag)


@router.delete(
    '/{package_id}/tag/{tag_instance_id}',
)
async def untag_package(
        package_id: str,
        tag_instance_id: str,
        auth: Authorized = Depends(UntagAuthorize(TagType.package)),
) -> None:
    """
    Remove a tag instance set by the calling user. The package status must be `editing`.
    Requires the scope associated with the tag.
    """
    await _untag_package(package_id, tag_instance_id, auth, True)


@router.delete(
    '/admin/{package_id}/tag/{tag_instance_id}',
)
async def admin_untag_package(
        package_id: str,
        tag_instance_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_ADMIN)),
) -> None:
    """
    Remove any tag instance from a package. Requires scope `odp.package:admin`.
    """
    await _untag_package(package_id, tag_instance_id, auth, False)


async def _untag_package(
        package_id: str,
        tag_instance_id: str,
        auth: Authorized,
        check_status: bool,
) -> None:
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    auth.enforce_constraint([package.provider_id])

    if check_status:
        ensure_status(package, PackageStatus.editing)

    await Tagger(TagType.package).delete_tag_instance(tag_instance_id, package, auth)


@router.post(
    '/{package_id}/submit',
)
async def submit_package(
        package_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_WRITE)),
) -> None:
    """
    Submit a provider-accessible package. Requires scope `odp.package:write`.
    """
    await _submit_package(package_id, auth)


@router.post(
    '/admin/{package_id}/submit',
)
async def admin_submit_package(
        package_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_ADMIN)),
) -> None:
    """
    Submit any package. Requires scope `odp.package:admin`.
    """
    await _submit_package(package_id, auth)


async def _submit_package(
        package_id: str,
        auth: Authorized,
):
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    auth.enforce_constraint([package.provider_id])

    ensure_status(package, PackageStatus.editing)

    tag_patch = []
    for package_tag in package.tags:
        tag_schema = schema_catalog.get_schema(URI(package_tag.tag.schema.uri))
        tag_schema_result = tag_schema.evaluate(JSON(package_tag.data))
        tag_patch += tag_schema_result.output('translation-patch', scheme=package.schema_id)

    _schema = Session.get(Schema, (package.schema_id, package.schema_type))
    metadata_schema = schema_catalog.get_schema(URI(_schema.uri))
    metadata_template = schema_catalog.load_json(URI(_schema.template_uri))
    metadata = JSONPatch(*tag_patch).evaluate(metadata_template, resolve_array_inserts=True)
    remove_empty_children(metadata)

    package.metadata_ = metadata
    package.validity = await get_metadata_validity(package.metadata_, metadata_schema)
    package.status = PackageStatus.submitted
    package.timestamp = (timestamp := datetime.now(timezone.utc))
    package.save()

    create_audit_record(auth, package, timestamp, PackageCommand.submit)


@router.post(
    '/{package_id}/cancel',
)
async def cancel_package(
        package_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_WRITE)),
) -> None:
    """
    Cancel submission of a provider-accessible package. Requires scope `odp.package:write`.
    """
    await _cancel_package(package_id, auth)


@router.post(
    '/admin/{package_id}/cancel',
)
async def admin_cancel_package(
        package_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_ADMIN)),
) -> None:
    """
    Cancel submission of any package. Requires scope `odp.package:admin`.
    """
    await _cancel_package(package_id, auth)


async def _cancel_package(
        package_id: str,
        auth: Authorized,
):
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    auth.enforce_constraint([package.provider_id])

    ensure_status(package, PackageStatus.submitted)

    package.status = PackageStatus.editing
    package.metadata_ = None
    package.validity = None
    package.timestamp = (timestamp := datetime.now(timezone.utc))
    package.save()

    create_audit_record(auth, package, timestamp, PackageCommand.cancel)


@router.put(
    '/{package_id}/files/{path:path}',
    dependencies=[Depends(ArchiveAuthorize())],
)
async def upload_file(
        package_id: str,
        archive_id: str,
        path: str = Path(..., title='File path relative to the package root'),
        file: UploadFile = File(..., title='File data'),
        sha256: str = Query(..., title='SHA-256 checksum'),
        title: str = Query(None, title='Resource title'),
        description: str = Query(None, title='Resource description'),
        unpack: bool = Query(False, title='Unpack zipped file data'),
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_WRITE)),
) -> None:
    """
    Upload a file to an archive and add/unpack it into a package folder.

    By default, a single resource is created and associated with the package
    and the archive. If `unpack` is True, `file` is assumed to be zipped data
    and its contents are unpacked at the parent directory of `path`. For each
    unpacked file, a resource is created and similarly associated.

    Existing files are replaced.

    Requires scope `odp.package:write` along with the scope associated with
    the archive. The package status must be `editing`.
    """
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Package not found')

    ensure_status(package, PackageStatus.editing)

    await _upload_file(
        package_id, archive_id, path, file.file, sha256, title, description, unpack, auth,
    )


async def _upload_file(
        package_id: str,
        archive_id: str,
        path: str,
        file: BinaryIO,
        sha256: str,
        title: str | None,
        description: str | None,
        unpack: bool,
        auth: Authorized,
):
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Package not found')

    if not (archive := Session.get(Archive, archive_id)):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Archive not found')

    auth.enforce_constraint([package.provider_id])

    path = pathlib.Path(path)
    if path.is_absolute():
        raise HTTPException(HTTP_400_BAD_REQUEST, 'path must be relative')

    for part in path.parts:
        if part != secure_filename(part):
            raise HTTPException(HTTP_400_BAD_REQUEST, 'invalid path')

    archive_adapter = ArchiveAdapter.get_instance(archive)
    archive_resource_path = f'{package.key}/{path}'
    try:
        file_info_list = await archive_adapter.put(
            archive_resource_path, file, sha256, unpack
        )
    except ArchiveError as e:
        raise HTTPException(e.status_code, e.error_detail) from e
    except NotImplementedError:
        raise HTTPException(HTTP_405_METHOD_NOT_ALLOWED, f'Operation not supported for {archive.id}')

    for file_info in file_info_list:
        archive_resource_path = file_info.path
        resource_path = file_info.path.removeprefix(f'{package.key}/')

        if not (resource := Session.execute(
                select(Resource)
                        .where(Resource.package_id == package_id)
                        .where(Resource.path == resource_path)
        ).scalar_one_or_none()):
            resource = Resource(
                package_id=package_id,
                path=resource_path,
            )

        resource.mimetype = mimetypes.guess_type(file_info.path, strict=False)[0]
        resource.size = file_info.size
        resource.hash = file_info.sha256
        resource.hash_algorithm = HashAlgorithm.sha256
        resource.title = title
        resource.description = description
        resource.status = ResourceStatus.active
        resource.timestamp = (timestamp := datetime.now(timezone.utc))
        resource.save()

        if not (archive_resource := Session.get(ArchiveResource, (archive_id, resource.id))):
            archive_resource = ArchiveResource(
                archive_id=archive.id,
                resource_id=resource.id,
            )

        archive_resource.path = archive_resource_path
        archive_resource.status = ArchiveResourceStatus.valid
        archive_resource.timestamp = timestamp
        archive_resource.save()

        # TODO: what about existing archive_resource records for other archives?


@router.get(
    '/{package_id}/files/{resource_id}',
    dependencies=[Depends(ArchiveAuthorize())],
)
async def download_file(
        package_id: str,
        resource_id: str,
        archive_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_READ)),
) -> StreamingResponse:
    """
    Download a package file from an archive. Requires scope `odp.package:read`
    along with the scope associated with the archive.
    """
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Package not found')

    auth.enforce_constraint([package.provider_id])

    if (
            not (resource := Session.get(Resource, resource_id)) or
            resource.package_id != package_id or
            resource.status != ResourceStatus.active
    ):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Resource not found')

    if not (archive := Session.get(Archive, archive_id)):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Archive not found')

    if not (archive_resource := Session.get(ArchiveResource, (archive_id, resource_id))):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Resource not found in archive')

    if archive_resource.status != ArchiveResourceStatus.valid:
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, f'Archived resource is {archive_resource.status}')

    archive_adapter = ArchiveAdapter.get_instance(archive)
    archive_response = await archive_adapter.get(archive_resource.path)

    if not isinstance(archive_response, ArchiveFileResponse):
        raise HTTPException(HTTP_422_UNPROCESSABLE_ENTITY, 'Resource is not a file')

    filename = pathlib.Path(resource.path).name
    return StreamingResponse(
        archive_response.file,
        media_type=resource.mimetype,
        headers={
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
    )


@router.delete(
    '/{package_id}/files/{resource_id}',
)
async def delete_file(
        package_id: str,
        resource_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.PACKAGE_WRITE)),
) -> None:
    """Delete a file.

    Updates the resource status to `delete_pending`; actual file deletions
    are performed by a background service.

    Requires scope `odp.package:write`. The package status must be `editing`.
    """
    if not (package := Session.get(Package, package_id)):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Package not found')

    auth.enforce_constraint([package.provider_id])
    ensure_status(package, PackageStatus.editing)

    if not (resource := Session.get(Resource, resource_id)):
        raise HTTPException(HTTP_404_NOT_FOUND, 'Resource not found')

    if resource.package_id != package_id:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Resource not found')

    if resource.status != ResourceStatus.delete_pending:
        resource.status = ResourceStatus.delete_pending
        resource.timestamp = datetime.now(timezone.utc)
        resource.save()
