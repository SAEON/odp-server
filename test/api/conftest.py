import time

import pytest
from authlib.jose import JsonWebKey, JsonWebToken, KeySet
from starlette.testclient import TestClient

import odp.api.main
import odp.lib.auth
from odp.const.db import TagCardinality
from test.api import all_scopes, all_scopes_excluding


@pytest.fixture(scope='session')
def test_jwk():
    return JsonWebKey.generate_key('RSA', 2048, is_private=True, options={'kid': 'test-api-key'})


@pytest.fixture
def api(monkeypatch, test_jwk):
    key_set = KeySet([test_jwk])
    monkeypatch.setattr(odp.lib.auth, 'get_jwks', lambda auth_url=None, force_refresh=False: key_set)
    jwt = JsonWebToken(['RS256'])

    def scoped_client(scopes, create_scopes=True):
        now = int(time.time())
        scope_values = [s.value if hasattr(s, 'value') else str(s) for s in scopes]
        payload = {
            'sub': 'odp.test',
            'azp': 'odp.test',
            'client_id': 'odp.test',
            'preferred_username': 'service-account-odp.test',
            'roles': scope_values,
            'scope': ' '.join(['openid'] + scope_values),
            'iat': now,
            'exp': now + 3600,
        }
        access_token = jwt.encode({'alg': 'RS256', 'kid': 'test-api-key'}, payload, test_jwk).decode('utf-8')
        api_client = TestClient(app=odp.api.main.app)
        api_client.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}',
        }
        return api_client

    return scoped_client


@pytest.fixture(params=TagCardinality)
def tag_cardinality(request):
    """Use for parameterizing the range of tag cardinalities."""
    return request.param


@pytest.fixture(params=[1, 2, 3, 4])
def scopes(request):
    """Fixture for parameterizing the set of auth scopes
    to be associated with the API test client.

    The test function must be decorated to indicate the scope
    required by the API route::

        @pytest.mark.require_scope(ODPScope.CATALOG_READ)
    """
    scope = request.node.get_closest_marker('require_scope').args[0]

    if request.param == 1:
        return [scope]
    elif request.param == 2:
        return []
    elif request.param == 3:
        return all_scopes
    elif request.param == 4:
        return all_scopes_excluding(scope)
