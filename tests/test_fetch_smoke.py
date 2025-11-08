import os
from datetime import date, timedelta
from portopt.core.data import fetch_prices, align_and_clean

import pytest

skip_net = os.getenv("SKIP_NETWORK") == "1"

@pytest.mark.skipif(skip_net, reason="network disabled in CI")
def test_fetch_prices_smoke():
    start = date.today() - timedelta(days=60)
    end = date.today()
    prices = fetch_prices(["AAPL"], start, end, interval="Daily")
    prices = align_and_clean(prices)
    assert not prices.empty
    assert "AAPL" in prices.columns
