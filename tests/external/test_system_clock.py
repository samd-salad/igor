from datetime import datetime
from server.external.system_clock import SystemClock


def test_now_is_aware():
    n = SystemClock().now()
    assert isinstance(n, datetime)
    assert n.tzinfo is not None
