from functools import partial

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from starlette.status import HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_ENTITY

from odp.api.lib.auth import Authorize
from odp.api.lib.paging import Page, Paginator
from odp.api.models import UserModel, UserModelIn
from odp.const import ODPScope
from odp.db import Session
from odp.db.models import Role, User

router = APIRouter()


@router.get(
    '/',
    response_model=Page[UserModel],
    dependencies=[Depends(Authorize(ODPScope.USER_READ))],
)
async def list_users(
        paginator: Paginator = Depends(partial(Paginator, sort='email')),
):
    return paginator.paginate(
        select(User),
        lambda row: UserModel(
            id=row.User.id,
            email=row.User.email,
            active=row.User.active,
            verified=row.User.verified,
            name=row.User.name,
            picture=row.User.picture,
            role_ids=[role.id for role in row.User.roles],
        )
    )


@router.get(
    '/{user_id}',
    response_model=UserModel,
    dependencies=[Depends(Authorize(ODPScope.USER_READ))],
)
async def get_user(
        user_id: str,
):
    if not (user := Session.get(User, user_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    return UserModel(
        id=user.id,
        email=user.email,
        active=user.active,
        verified=user.verified,
        name=user.name,
        picture=user.picture,
        role_ids=[role.id for role in user.roles],
    )


@router.put(
    '/',
    dependencies=[Depends(Authorize(ODPScope.USER_ADMIN))],
)
async def update_user(
        user_in: UserModelIn,
):
    if not (user := Session.get(User, user_in.id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    user.active = user_in.active
    user.roles = [
        Session.get(Role, role_id)
        for role_id in user_in.role_ids
    ]
    user.save()


@router.delete(
    '/{user_id}',
    dependencies=[Depends(Authorize(ODPScope.USER_ADMIN))],
)
async def delete_user(
        user_id: str,
):
    if not (user := Session.get(User, user_id)):
        raise HTTPException(HTTP_404_NOT_FOUND)

    try:
        user.delete()
    except IntegrityError as e:
        raise HTTPException(
            HTTP_422_UNPROCESSABLE_ENTITY,
            'The user cannot be deleted due to associated tag instance data.',
        ) from e
