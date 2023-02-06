from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_mail import Message

from odp.identity import hydra_admin_api, mail
from odp.identity.forms import ProfileForm, ResetPasswordForm
from odp.identity.lib import (get_user_profile, password_complexity_description, update_user_password, update_user_profile, update_user_verified,
                              validate_auto_login, validate_email_verification, validate_password_reset)
from odp.identity.views import decode_token, encode_token
from odp.lib import exceptions as x

bp = Blueprint('account', __name__)


@bp.route('/verify-email')
def verify_email():
    """Target for email verification links.

    The token ensures that this view is only accessible from a verification email.
    """
    token = request.args.get('token')
    login_request, challenge, brand, params = decode_token(token, 'account.verify_email')

    email = params.get('email')
    try:
        user_id = validate_email_verification(email)
        update_user_verified(user_id, True)
        flash("Your email address has been verified.")

        complete_token = encode_token('account.verify_email_complete', challenge, brand, user_id=user_id)
        redirect_to = url_for('.verify_email_complete', token=complete_token)

    except x.ODPIdentityError as e:
        # any validation error => reject login
        redirect_to = hydra_admin_api.reject_login_request(challenge, e.error_code, e.error_description)

    return redirect(redirect_to)


@bp.route('/verify-email-complete', methods=('GET', 'POST'))
def verify_email_complete():
    """View for concluding the login with Hydra after verifying an email address.

    The token ensures that we can only get here from the verify email view.
    """
    token = request.args.get('token')
    login_request, challenge, brand, params = decode_token(token, 'account.verify_email_complete')

    client_id = login_request['client']['client_id']
    user_id = params.get('user_id')

    if request.method == 'POST':
        try:
            validate_auto_login(client_id, user_id)
            redirect_to = hydra_admin_api.accept_login_request(challenge, user_id)

        except x.ODPIdentityError as e:
            # any validation error => reject login
            redirect_to = hydra_admin_api.reject_login_request(challenge, e.error_code, e.error_description)

        return redirect(redirect_to)

    return render_template('verify_email_complete.html', token=token, brand=brand)


@bp.route('/profile', methods=('GET', 'POST'))
def profile():
    """View for updating user profile info."""

    token = request.args.get('token')
    login_request, challenge, brand, params = decode_token(token, 'account.profile')

    client_id = login_request['client']['client_id']
    user_id = params.get('user_id')
    user_info = get_user_profile(user_id)
    form = ProfileForm(**user_info)

    if request.method == 'POST':
        try:
            validate_auto_login(client_id, user_id)
            update_user_profile(user_id, **form.data)
            redirect_to = hydra_admin_api.accept_login_request(challenge, user_id)

        except x.ODPIdentityError as e:
            # any validation error => reject login
            redirect_to = hydra_admin_api.reject_login_request(challenge, e.error_code, e.error_description)

        return redirect(redirect_to)

    return render_template('profile.html', form=form, token=token, brand=brand)


@bp.route('/reset-password', methods=('GET', 'POST'))
def reset_password():
    """Target for password reset links.

    The token ensures that this view is only accessible from a password reset email.
    """
    token = request.args.get('token')
    login_request, challenge, brand, params = decode_token(token, 'account.reset_password')

    client_id = login_request['client']['client_id']
    form = ResetPasswordForm(request.form)
    email = params.get('email')

    if request.method == 'POST':
        if form.validate():
            password = form.password.data
            redirect_to = None
            try:
                user_id = validate_password_reset(client_id, email, password)
                update_user_password(user_id, password)
                flash("Your password has been changed.")

                complete_token = encode_token('account.reset_password_complete', challenge, brand, user_id=user_id)
                redirect_to = url_for('.reset_password_complete', token=complete_token)

            except x.ODPPasswordComplexityError:
                form.password.errors.append("The password does not meet the minimum complexity requirements.")
                flash(password_complexity_description(), category='info')

            except x.ODPIdentityError as e:
                # any other validation error => reject login
                redirect_to = hydra_admin_api.reject_login_request(challenge, e.error_code, e.error_description)

            if redirect_to:
                return redirect(redirect_to)

    return render_template('reset_password.html', form=form, token=token, brand=brand)


@bp.route('/reset-password-complete', methods=('GET', 'POST'))
def reset_password_complete():
    """View for concluding the login with Hydra after changing a password.

    The token ensures that we can only get here from the reset password view.
    """
    token = request.args.get('token')
    login_request, challenge, brand, params = decode_token(token, 'account.reset_password_complete')

    client_id = login_request['client']['client_id']
    user_id = params.get('user_id')

    if request.method == 'POST':
        try:
            validate_auto_login(client_id, user_id)
            redirect_to = hydra_admin_api.accept_login_request(challenge, user_id)

        except x.ODPIdentityError as e:
            # any validation error => reject login
            redirect_to = hydra_admin_api.reject_login_request(challenge, e.error_code, e.error_description)

        return redirect(redirect_to)

    return render_template('reset_password_complete.html', token=token, brand=brand)


def send_verification_email(email, name, challenge, brand):
    """Send an email address verification email.

    :param email: the email address to be verified
    :param name: the user's full name
    :param challenge: the Hydra login challenge
    :param brand: branding identifier
    """
    try:
        token = encode_token('account.verify_email', challenge, brand, email=email)
        context = {
            'url': url_for('account.verify_email', token=token, _external=True),
            'name': name or '',
            'brand': brand,
        }
        msg = Message(
            subject=render_template('email/verify_email_subject.txt', **context),
            body=render_template('email/verify_email.txt', **context),
            html=render_template('email/verify_email.html', **context),
            recipients=[email],
            sender=("SAEON", "noreply@saeon.nrf.ac.za"),
            reply_to=("SAEON", "noreply@saeon.nrf.ac.za"),
        )
        mail.send(msg)
        flash("An email verification link has been sent to your email address.")
    except Exception as e:
        current_app.logger.error("Error sending verification email to {}: {}".format(email, e))
        flash("There was a problem sending the verification email.", category='error')


def send_password_reset_email(email, name, challenge, brand):
    """Send a password reset email.

    :param email: the email address
    :param name: the user's full name
    :param challenge: the Hydra login challenge
    :param brand: branding identifier
    """
    try:
        token = encode_token('account.reset_password', challenge, brand, email=email)
        context = {
            'url': url_for('account.reset_password', token=token, _external=True),
            'name': name or '',
            'brand': brand,
        }
        msg = Message(
            subject=render_template('email/reset_password_subject.txt', **context),
            body=render_template('email/reset_password.txt', **context),
            html=render_template('email/reset_password.html', **context),
            recipients=[email],
            sender=("SAEON", "noreply@saeon.nrf.ac.za"),
            reply_to=("SAEON", "noreply@saeon.nrf.ac.za"),
        )
        mail.send(msg)
        flash("A password reset link has been sent to your email address.")
    except Exception as e:
        current_app.logger.error("Error sending password reset email to {}: {}".format(email, e))
        flash("There was a problem sending the password reset email.", category='error')
