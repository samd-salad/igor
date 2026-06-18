from server.ha_io._internal.auth import check_token
from server.ha_io._internal.rate_limit import RateLimiter


def test_check_token_passes_when_env_unset(monkeypatch):
    monkeypatch.delenv("IGOR_API_TOKEN", raising=False)
    assert check_token(provided=None) is True


def test_check_token_fails_when_mismatch(monkeypatch):
    monkeypatch.setenv("IGOR_API_TOKEN", "secret")
    assert check_token(provided="wrong") is False
    assert check_token(provided="secret") is True


def test_rate_limiter_blocks_after_max():
    rl = RateLimiter(max_requests=2, window_seconds=60.0)
    assert rl.is_allowed("ip-1") is True
    assert rl.is_allowed("ip-1") is True
    assert rl.is_allowed("ip-1") is False
    assert rl.is_allowed("ip-2") is True
