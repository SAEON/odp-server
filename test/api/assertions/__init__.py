from datetime import datetime, timedelta, timezone


def assert_ok_null(response):
    assert response.status_code == 200
    assert response.json() is None


def assert_forbidden(response):
    assert response.status_code == 403
    assert response.json() == {'detail': 'Forbidden'}


def assert_not_found(response, message='Not Found'):
    assert response.status_code == 404
    assert response.json() == {'detail': message}


def assert_method_not_allowed(response):
    assert response.status_code == 405
    assert response.json() == {'detail': 'Method Not Allowed'}


def assert_conflict(response, message):
    assert response.status_code == 409
    assert response.json() == {'detail': message}


def assert_unprocessable(response, message=None, **kwargs):
    # kwargs are key-value pairs expected within 'detail'
    assert response.status_code == 422
    error_detail = response.json()['detail']
    if message is not None:
        assert error_detail == message
    for k, v in kwargs.items():
        assert error_detail[k] == v


def assert_new_timestamp(timestamp):
    # 1 hour is a bit lenient, but handy for debugging
    assert (now := datetime.now(timezone.utc)) - timedelta(minutes=60) < timestamp < now


def assert_redirect(response, url):
    assert response.is_redirect
    assert response.next_request.url == url
