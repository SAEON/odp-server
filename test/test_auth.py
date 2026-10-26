import time

import pytest
from authlib.jose import JsonWebKey, JsonWebToken, KeySet

import odp.lib.auth
from odp.lib.auth import (
    Authorized,
    TokenVerificationError,
    decode_and_verify_token,
    resolve_authorized,
)


@pytest.fixture
def test_jwk():
    return JsonWebKey.generate_key('RSA', 2048, is_private=True, options={'kid': 'test-key-1'})


@pytest.fixture
def mock_jwks(monkeypatch, test_jwk):
    key_set = KeySet([test_jwk])
    monkeypatch.setattr(odp.lib.auth, 'get_jwks', lambda auth_url=None, force_refresh=False: key_set)
    return test_jwk


def _sign_token(jwk, payload: dict) -> str:
    jwt = JsonWebToken(['RS256'])
    header = {'alg': 'RS256', 'kid': 'test-key-1'}
    return jwt.encode(header, payload, jwk).decode('utf-8')


def test_decode_and_verify_valid_token(mock_jwks):
    now = int(time.time())
    payload = {
        'sub': 'user-123',
        'azp': 'ODP.Web.UI',
        'name': 'Jane Researcher',
        'email': 'jane@saeon.ac.za',
        'roles': ['odp.record:read', 'odp.submission:write'],
        'scope': 'openid odp.record:read',
        'iat': now,
        'exp': now + 300,
    }
    token = _sign_token(mock_jwks, payload)
    claims = decode_and_verify_token(token)
    assert claims['sub'] == 'user-123'
    assert claims['email'] == 'jane@saeon.ac.za'


def test_decode_and_verify_expired_token(mock_jwks):
    now = int(time.time())
    payload = {
        'sub': 'user-123',
        'iat': now - 600,
        'exp': now - 300,
    }
    token = _sign_token(mock_jwks, payload)
    with pytest.raises(TokenVerificationError):
        decode_and_verify_token(token)


def test_decode_and_verify_invalid_token(mock_jwks):
    with pytest.raises(TokenVerificationError):
        decode_and_verify_token('not-a-valid-jwt')


def test_resolve_authorized_human_user_intersection():
    """Effective scopes should be the intersection of user roles and client scopes."""
    claims = {
        'sub': 'user-uuid-1',
        'azp': 'ODP.Web.UI',
        'name': 'Dylan Pivo',
        'email': 'dylanpivo@gmail.com',
        'preferred_username': 'dylanpivo@gmail.com',
        'roles': ['odp.record:read', 'odp.record:admin', 'odp.token:read'],
        'scope': 'openid odp.record:read odp.token:read odp.submission:write',
    }
    auth = resolve_authorized(claims)
    assert auth == Authorized(
        client_id='ODP.Web.UI',
        user_id='user-uuid-1',
        user_name='Dylan Pivo',
        scopes={'odp.record:read', 'odp.token:read'},
    )


def test_resolve_authorized_service_account():
    """Service account tokens should resolve user_id=None and user_name=None."""
    claims = {
        'sub': 'sa-uuid-1',
        'azp': 'ODP.Web.CI',
        'client_id': 'ODP.Web.CI',
        'preferred_username': 'service-account-odp.web.ci',
        'roles': ['odp.catalog:read', 'odp.catalog:search'],
        'scope': 'openid odp.catalog:read odp.catalog:search',
    }
    auth = resolve_authorized(claims)
    assert auth == Authorized(
        client_id='ODP.Web.CI',
        user_id=None,
        user_name=None,
        scopes={'odp.catalog:read', 'odp.catalog:search'},
    )
