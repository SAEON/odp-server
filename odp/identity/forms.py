from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import email, equal_to, input_required


class SignupForm(FlaskForm):
    name = StringField(
        label='Full name',
        validators=[input_required()],
    )
    email = StringField(
        label='Email address',
        filters=[lambda s: s.lower() if s else s],
        validators=[input_required(), email()],
    )
    password = PasswordField(
        label='Password',
        validators=[input_required(), equal_to('confirm_password', "The passwords do not match")],
    )
    confirm_password = PasswordField(
        label='Confirm password',
        validators=[input_required()],
    )
    submit = SubmitField(
        label='Sign up',
    )


class LoginForm(FlaskForm):
    email = StringField(
        label='Email address',
        filters=[lambda s: s.lower() if s else s],
        validators=[input_required(), email()],
    )
    password = PasswordField(
        label='Password',
        validators=[input_required()],
    )
    submit = SubmitField(
        label='Log in',
    )


class VerifyEmailForm(FlaskForm):
    submit = SubmitField(
        label='Re-send verification email',
    )


class ForgotPasswordForm(FlaskForm):
    email = StringField(
        label='Email address',
        filters=[lambda s: s.lower() if s else s],
        validators=[input_required(), email()],
    )
    submit = SubmitField(
        label='Send reset link',
    )


class ResetPasswordForm(FlaskForm):
    password = PasswordField(
        label='Password',
        validators=[input_required(), equal_to('confirm_password', "The passwords do not match")],
    )
    confirm_password = PasswordField(
        label='Confirm password',
        validators=[input_required()],
    )
    submit = SubmitField(
        label='Set new password',
    )


class AutoLoginForm(FlaskForm):
    submit = SubmitField(
        label='Log in',
    )


class ProfileForm(FlaskForm):
    name = StringField(label='Full name')
    picture = StringField(label='Photo URL')
    submit = SubmitField(label='Log in')
