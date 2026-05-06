from nurai.api.middleware import FixedWindowRateLimiter


def test_rate_limiter_allows_within_window() -> None:
    limiter = FixedWindowRateLimiter(requests=2, window_seconds=60)

    assert limiter.allow("client") is True
    assert limiter.allow("client") is True
    assert limiter.allow("client") is False


def test_rate_limiter_cleanup_drops_expired_keys() -> None:
    limiter = FixedWindowRateLimiter(requests=5, window_seconds=0)

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True
    assert limiter.tracked_keys() == 2

    limiter.cleanup()

    assert limiter.tracked_keys() == 0


def test_rate_limiter_cleanup_keeps_active_keys() -> None:
    limiter = FixedWindowRateLimiter(requests=5, window_seconds=60)

    assert limiter.allow("client-a") is True

    limiter.cleanup()

    assert limiter.tracked_keys() == 1
