from flask import abort, current_app
from itsdangerous import BadData, JSONWebSignatureSerializer

from odp.identity import hydra_admin


def init_app(app):
    from . import hydra_workflow, login, signup, account, google, status

    app.register_blueprint(hydra_workflow.bp, url_prefix='/hydra')
    app.register_blueprint(login.bp, url_prefix='/login')
    app.register_blueprint(signup.bp, url_prefix='/signup')
    app.register_blueprint(account.bp, url_prefix='/account')
    app.register_blueprint(status.bp, url_prefix='/status')
    app.register_blueprint(google.bp, url_prefix='/google')


def encode_token(scope: str, challenge: str, brand: str, **params):
    """
    Create a JWS token for accessing application views (other than the Hydra workflow views)
    which may only be accessed within the context of the Hydra login workflow.

    `scope` restricts the usage of tokens and allows us to control the sequence in which
    views may be exposed. It enables a given token to be re-used across multiple views, but
    only where those views expect the same token scope. So we might, for example, allow a
    user to switch between login and signup views with the same token, but prevent them
    from copying a password reset token and passing it to the email verification view.

    :param scope: the scope for which the token is valid
    :param challenge: the Hydra login challenge
    :param brand: UI branding identifier
    :param params: any additional params to pass to the view
    :return: a JSON Web Signature
    """
    serializer = JSONWebSignatureSerializer(current_app.secret_key, salt=scope)
    params.update({'challenge': challenge, 'brand': brand})
    token = serializer.dumps(params)
    return token


def decode_token(token: str, scope: str):
    """
    Decode and validate a JWS token received by a view, and return the Hydra login
    request dict and login challenge, along with the UI brand identifier and any
    additional params.

    :param token: the token to decode
    :param scope: the scope for which the token is valid
    :return: tuple(login_request: dict, challenge: str, brand: str, params: dict)
    :raises HydraAdminError: if the encoded login challenge is invalid
    """
    if not token:
        abort(403)  # HTTP 403 Forbidden

    try:
        serializer = JSONWebSignatureSerializer(current_app.secret_key, salt=scope)
        params = serializer.loads(token)
        challenge = params.pop('challenge', '')
        brand = params.pop('brand', '')
        login_request = hydra_admin.get_login_request(challenge)
        return login_request, challenge, brand, params

    except BadData:
        abort(403)  # HTTP 403 Forbidden


def hydra_error_page(e):
    """
    Requests to the Hydra admin API are critical to the login, consent and logout flows.
    If anything is wrong with a response from Hydra, we abort.

    :param e: the HydraAdminError exception
    """
    current_app.logger.critical(
        "Hydra %d error for %s %s: %s",
        e.status_code, e.method, e.endpoint, e.error_detail
    )
    abort(500)
