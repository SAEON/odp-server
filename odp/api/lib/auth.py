from fastapi import HTTPException
from fastapi.openapi.models import OAuth2, OAuthFlowClientCredentials, OAuthFlows
from fastapi.security.base import SecurityBase
from fastapi.security.utils import get_authorization_scheme_param
from sqlalchemy import select
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND

from odp.api.models import TagInstanceModelIn
from odp.config import config
from odp.const import ODPScope
from odp.const.db import TagType
from odp.db import Session
from odp.db.models import CollectionTag, RecordTag, Tag, Vocabulary
from odp.lib.auth import (
    Authorized,
    AuthServiceUnavailableError,
    TokenVerificationError,
    decode_and_verify_token,
    resolve_authorized,
)

auth_url = str(config.AUTH.URL).rstrip('/')


def _authorize_request(request: Request, required_scope_id: str) -> Authorized:
    auth_header = request.headers.get('Authorization')
    scheme, access_token = get_authorization_scheme_param(auth_header)
    if not auth_header or scheme.lower() != 'bearer':
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            headers={'WWW-Authenticate': 'Bearer'},
        )

    try:
        claims = decode_and_verify_token(access_token, auth_url=auth_url)
    except AuthServiceUnavailableError as e:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail='Authentication service unavailable') from e
    except TokenVerificationError as e:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            headers={'WWW-Authenticate': 'Bearer'},
        ) from e

    authorized = resolve_authorized(claims)
    if required_scope_id not in authorized.scopes:
        raise HTTPException(HTTP_403_FORBIDDEN)

    return authorized


class BaseAuthorize(SecurityBase):

    def __init__(self):
        # OpenAPI docs / Swagger auth
        self.scheme_name = 'ODP API Authorization'
        self.model = OAuth2(flows=OAuthFlows(clientCredentials=OAuthFlowClientCredentials(
            tokenUrl=f'{auth_url}/protocol/openid-connect/token',
            scopes={s.value: s.value for s in ODPScope},
        )))


class Authorize(BaseAuthorize):
    def __init__(self, scope: ODPScope):
        super().__init__()
        self.scope_id = scope.value

    async def __call__(self, request: Request) -> Authorized:
        return _authorize_request(request, self.scope_id)


class TagAuthorize(BaseAuthorize):
    async def __call__(self, request: Request, tag_instance_in: TagInstanceModelIn) -> Authorized:
        if not (tag_scope_id := Session.execute(
                select(Tag.scope_id).
                where(Tag.id == tag_instance_in.tag_id)
        ).scalar_one_or_none()):
            raise HTTPException(HTTP_404_NOT_FOUND)

        return _authorize_request(request, tag_scope_id)


class UntagAuthorize(BaseAuthorize):
    def __init__(self, tag_type: TagType):
        super().__init__()
        self.tag_type = tag_type

    async def __call__(self, request: Request, tag_instance_id: str) -> Authorized:
        if self.tag_type == TagType.record:
            stmt = (
                select(Tag.scope_id).
                join(RecordTag).
                where(RecordTag.id == tag_instance_id)
            )
        elif self.tag_type == TagType.collection:
            stmt = (
                select(Tag.scope_id).
                join(CollectionTag).
                where(CollectionTag.id == tag_instance_id)
            )
        else:
            assert False

        if not (tag_scope_id := Session.execute(stmt).scalar_one_or_none()):
            raise HTTPException(HTTP_404_NOT_FOUND)

        return _authorize_request(request, tag_scope_id)


class VocabularyAuthorize(BaseAuthorize):
    async def __call__(self, request: Request, vocabulary_id: str) -> Authorized:
        if not (vocabulary_scope_id := Session.execute(
                select(Vocabulary.scope_id).
                where(Vocabulary.id == vocabulary_id)
        ).scalar_one_or_none()):
            raise HTTPException(HTTP_404_NOT_FOUND)

        return _authorize_request(request, vocabulary_scope_id)
