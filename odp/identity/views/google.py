from authlib.integrations.base_client.errors import OAuthError
from flask import Blueprint, redirect, request, url_for

from odp.identity import google_oauth2, hydra_admin_api
from odp.identity.lib import create_user_account, update_user_profile, update_user_verified, validate_google_login
from odp.identity.views import decode_token, encode_token
from odp.lib import exceptions as x

bp = Blueprint('google', __name__)


@bp.route('/authorize', methods=('POST',))
def authorize():
    """View for initiating the Google OAuth2 flow.

    This enables us to authenticate a user via Google, and enables the
    user to authorize the ODP to access their Google profile.

    The token ensures that we can only access this view in the context
    of the Hydra login workflow.
    """
    token = request.args.get('token')
    login_request, challenge, brand, params = decode_token(token, 'login')
    authorized_token = encode_token('google.authorized', challenge, brand)
    redirect_uri = url_for('.authorized', _external=True)
    return google_oauth2.google.authorize_redirect(redirect_uri, state=authorized_token)


@bp.route('/authorized')
def authorized():
    """Callback from Google.

    Create an account for the user if one does not already exist,
    pull their profile info from Google, and log them in.

    The token in the 'state' param ensures that this view can only be
    accessed in the context of the Google OAuth2 flow, initiated from
    our Google authorize view.
    """
    token = request.args.get('state')
    login_request, challenge, brand, params = decode_token(token, 'google.authorized')
    try:
        try:
            google_token = google_oauth2.google.authorize_access_token()
            userinfo = google_token.pop('userinfo')
            email = userinfo['email']
            email_verified = userinfo.get('email_verified')

            if not email_verified:
                raise x.ODPEmailNotVerified

        except (OAuthError, KeyError):
            raise x.ODPGoogleAuthError

        try:
            user_id = validate_google_login(email)
        except x.ODPUserNotFound:
            user_id = create_user_account(email)

        update_user_verified(user_id, True)
        update_user_profile(user_id, **userinfo)
        redirect_to = hydra_admin_api.accept_login_request(challenge, user_id)

    except x.ODPIdentityError as e:
        redirect_to = hydra_admin_api.reject_login_request(challenge, e.error_code, e.error_description)

    return redirect(redirect_to)
