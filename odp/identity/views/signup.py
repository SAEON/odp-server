from flask import Blueprint, flash, redirect, render_template, request, url_for

from odp.config import config
from odp.identity import hydra_admin_api
from odp.identity.forms import SignupForm
from odp.identity.lib import create_user_account, password_complexity_description
from odp.identity.views import decode_token, encode_token, hydra_error_page
from odp.identity.views.account import send_verification_email
from odp.lib import exceptions as x

bp = Blueprint('signup', __name__)


@bp.route('/', methods=('GET', 'POST'))
def signup():
    """User signup view.

    The token ensures that we can only access this view in the context
    of the Hydra login workflow.
    """
    token = request.args.get('token')
    try:
        # the token scope 'login' here is correct - it enables us to easily
        # switch between login and signup using the same token
        login_request, challenge, brand, params = decode_token(token, 'login')

        form = SignupForm(request.form)
        try:
            if request.method == 'GET':
                # if the user is already authenticated with Hydra, their user id is
                # associated with the login challenge; we cannot then associate a new
                # user id with the same login challenge
                authenticated = login_request['skip']
                if authenticated:
                    raise x.ODPSignupAuthenticatedUser

            else:  # POST
                if form.validate():
                    email = form.email.data
                    password = form.password.data
                    name = form.name.data
                    try:
                        create_user_account(email, password, name)

                        # the signup (and login) is completed via email verification
                        send_verification_email(email, name, challenge, brand)
                        verify_token = encode_token('signup.verify', challenge, brand, email=email, name=name)
                        redirect_to = url_for('.verify', token=verify_token)

                        return redirect(redirect_to)

                    except x.ODPEmailInUse:
                        form.email.errors.append("The email address is already associated with a user account.")

                    except x.ODPPasswordComplexityError:
                        form.password.errors.append("The password does not meet the minimum complexity requirements.")
                        flash(password_complexity_description(), category='info')

            return render_template('signup.html', form=form, token=token, brand=brand, enable_google=config.GOOGLE.ENABLE)

        except x.ODPIdentityError as e:
            # any other validation error (e.g. user already authenticated) => reject login
            redirect_to = hydra_admin_api.reject_login_request(challenge, e.error_code, e.error_description)
            return redirect(redirect_to)

    except x.HydraAdminError as e:
        return hydra_error_page(e)


@bp.route('/verify', methods=('GET', 'POST'))
def verify():
    """View for sending a verification email.

    The token ensures that we can only get here from the user signup view.
    """
    token = request.args.get('token')
    try:
        login_request, challenge, brand, params = decode_token(token, 'signup.verify')

        email = params.get('email')
        name = params.get('name')

        if request.method == 'POST':
            send_verification_email(email, name, challenge, brand)

        return render_template('signup_verify.html', token=token, brand=brand)

    except x.HydraAdminError as e:
        return hydra_error_page(e)
