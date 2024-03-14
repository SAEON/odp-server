from datetime import datetime, timezone
from functools import partial

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from starlette.status import HTTP_404_NOT_FOUND, HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_ENTITY

from odp.api.lib.auth import Authorize, Authorized
from odp.api.lib.paging import Page, Paginator
from odp.api.models import ProviderAuditModel, ProviderModel, ProviderModelIn
from odp.const import ODPScope
from odp.const.db import AuditCommand
from odp.db import Session
from odp.db.models import Provider, ProviderAudit, User

router = APIRouter()


def output_provider_model(provider: Provider) -> ProviderModel:
    return ProviderModel(
        id=provider.id,
        key=provider.key,
        name=provider.name,
        collection_keys={
            collection.key: collection.id
            for collection in provider.collections
        },
        user_ids={
            user.id: user.name
            for user in provider.users
        },
        timestamp=provider.timestamp.isoformat(),
    )


def output_audit_model(row) -> ProviderAuditModel:
    return ProviderAuditModel(
        table='provider',
        tag_id=None,
        audit_id=row.ProviderAudit.id,
        client_id=row.ProviderAudit.client_id,
        user_id=row.ProviderAudit.user_id,
        user_name=row.user_name,
        command=row.ProviderAudit.command,
        timestamp=row.ProviderAudit.timestamp.isoformat(),
        provider_id=row.ProviderAudit._id,
        provider_key=row.ProviderAudit._key,
        provider_name=row.ProviderAudit._name,
        provider_users=row.ProviderAudit._users,
    )


def create_audit_record(
        auth: Authorized,
        provider: Provider,
        timestamp: datetime,
        command: AuditCommand,
) -> None:
    ProviderAudit(
        client_id=auth.client_id,
        user_id=auth.user_id,
        command=command,
        timestamp=timestamp,
        _id=provider.id,
        _key=provider.key,
        _name=provider.name,
        _users=[user.id for user in provider.users],
    ).save()


@router.get(
    '/',
    response_model=Page[ProviderModel],
    dependencies=[Depends(Authorize(ODPScope.PROVIDER_READ))],
)
async def list_providers(
        paginator: Paginator = Depends(partial(Paginator, sort='key')),
):
    return paginator.paginate(
        select(Provider),
        lambda row: output_provider_model(row.Provider),
    )


@router.get(
    '/{provider_id}',
    response_model=ProviderModel,
    dependencies=[Depends(Authorize(ODPScope.PROVIDER_READ))],
)
async def get_provider(
        provider_id: str,
):
    if not (provider := Session.get(Provider, provider_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_provider_model(provider)


@router.post(
    '/',
    response_model=ProviderModel,
)
async def create_provider(
        provider_in: ProviderModelIn,
        auth: Authorized = Depends(Authorize(ODPScope.PROVIDER_ADMIN)),
):
    if Session.execute(
            select(Provider).
            where(Provider.key == provider_in.key)
    ).first() is not None:
        raise HTTPException(HTTP_409_CONFLICT, 'Provider key is already in use')

    provider = Provider(
        key=provider_in.key,
        name=provider_in.name,
        users=[
            Session.get(User, user_id)
            for user_id in provider_in.user_ids
        ],
        timestamp=(timestamp := datetime.now(timezone.utc)),
    )
    provider.save()
    create_audit_record(auth, provider, timestamp, AuditCommand.insert)

    return output_provider_model(provider)


@router.put(
    '/{provider_id}',
)
async def update_provider(
        provider_id: str,
        provider_in: ProviderModelIn,
        auth: Authorized = Depends(Authorize(ODPScope.PROVIDER_ADMIN)),
):
    if not (provider := Session.get(Provider, provider_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    if Session.execute(
            select(Provider).
            where(Provider.id != provider_id).
            where(Provider.key == provider_in.key)
    ).first() is not None:
        raise HTTPException(HTTP_409_CONFLICT, 'Provider key is already in use')

    if (
            provider.key != provider_in.key or
            provider.name != provider_in.name
    ):
        provider.key = provider_in.key
        provider.name = provider_in.name
        provider.users = [
            Session.get(User, user_id)
            for user_id in provider_in.user_ids
        ]
        provider.timestamp = (timestamp := datetime.now(timezone.utc))
        provider.save()
        create_audit_record(auth, provider, timestamp, AuditCommand.update)


@router.delete(
    '/{provider_id}',
)
async def delete_provider(
        provider_id: str,
        auth: Authorized = Depends(Authorize(ODPScope.PROVIDER_ADMIN)),
):
    if not (provider := Session.get(Provider, provider_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    create_audit_record(auth, provider, datetime.now(timezone.utc), AuditCommand.delete)

    try:
        provider.delete()
    except IntegrityError as e:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY,
            'A provider with non-empty collections cannot be deleted.',
        ) from e


@router.get(
    '/{provider_id}/audit',
    response_model=Page[ProviderAuditModel],
    dependencies=[Depends(Authorize(ODPScope.PROVIDER_READ))],
)
async def get_provider_audit_log(
        provider_id: str,
        paginator: Paginator = Depends(partial(Paginator, sort='timestamp')),
):
    stmt = (
        select(ProviderAudit, User.name.label('user_name')).
        outerjoin(User, ProviderAudit.user_id == User.id).
        where(ProviderAudit._id == provider_id)
    )

    return paginator.paginate(
        stmt,
        lambda row: output_audit_model(row),
    )


@router.get(
    '/{provider_id}/audit/{audit_id}',
    response_model=ProviderAuditModel,
    dependencies=[Depends(Authorize(ODPScope.PROVIDER_READ))],
)
async def get_provider_audit_detail(
        provider_id: str,
        audit_id: int,
):
    if not (row := Session.execute(
            select(ProviderAudit, User.name.label('user_name')).
            outerjoin(User, ProviderAudit.user_id == User.id).
            where(ProviderAudit.id == audit_id).
            where(ProviderAudit._id == provider_id)
    ).one_or_none()):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return output_audit_model(row)
