import pytest
from authlib.integrations.requests_client import OAuth2Session
from starlette.testclient import TestClient

import odp.api.main
from odp.config import config
from odp.const import ODPScope
from odp.const.db import TagCardinality
from odp.db import Session
from odp.db.models import Scope
from odp.lib.hydra import HydraAdminAPI
from test.api import CollectionAuth, all_scopes, all_scopes_excluding
from test.factories import ClientFactory, ScopeFactory

hydra_admin_url = config.HYDRA.ADMIN.URL
hydra_public_url = config.HYDRA.PUBLIC.URL


@pytest.fixture
def api():
    def scoped_client(scopes, collections=None, create_scopes=True):
        if create_scopes:
            scope_objects = [ScopeFactory(id=s.value, type='odp') for s in scopes]
        else:
            scope_objects = [Session.get(Scope, (s.value, 'odp')) for s in scopes]

        ClientFactory(
            id='odp.test',
            scopes=scope_objects,
            collection_specific=collections is not None,
            collections=collections,
        )
        token = OAuth2Session(
            client_id='odp.test',
            client_secret='secret',
            scope=' '.join(s.value for s in ODPScope),
        ).fetch_token(
            f'{hydra_public_url}/oauth2/token',
            grant_type='client_credentials',
            timeout=1.0,
        )
        api_client = TestClient(app=odp.api.main.app)
        api_client.headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer ' + token['access_token'],
        }
        return api_client

    return scoped_client


@pytest.fixture(scope='session')
def hydra_admin_api():
    return HydraAdminAPI(hydra_admin_url)


@pytest.fixture(params=CollectionAuth)
def collection_auth(request):
    """Use for parameterizing the three possible logic branches
    involving collection-specific authorization."""
    return request.param


@pytest.fixture(params=TagCardinality)
def tag_cardinality(request):
    """Use for parameterizing the range of tag cardinalities."""
    return request.param


@pytest.fixture(params=[1, 2, 3, 4])
def scopes(request):
    """Fixture for parameterizing the set of auth scopes
    to be associated with the API test client.

    The test function must be decorated to indicated the scope
    required by the API route::

        @pytest.mark.require_scope(ODPScope.CATALOG_READ)

    This has the same effect as parameterizing the test function
    as follows::

        @pytest.mark.parametrize('scopes', [
            [ODPScope.CATALOG_READ],
            [],
            all_scopes,
            all_scopes_excluding(ODPScope.CATALOG_READ),
        ])

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
