from fastapi import APIRouter, Depends

from odp.api.lib.auth import Authorize, Authorized
from odp.api.models import AccessTokenModel
from odp.const import ODPScope

router = APIRouter()


@router.get(
    '/',
    response_model=AccessTokenModel,
)
async def get_access_token_data(
        auth: Authorized = Depends(Authorize(ODPScope.TOKEN_READ)),
):
    return AccessTokenModel(
        client_id=auth.client_id,
        user_id=auth.user_id,
        permissions={scope_id: '*' for scope_id in auth.scopes},
    )

