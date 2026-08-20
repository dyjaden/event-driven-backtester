import pytest

from backtester.costs import (
    HalfSpreadSlippage,
    PerShareCommission,
    ZeroCommission,
    ZeroSlippage,
)


def test_commission_is_positive_on_a_sell():
    """quantity is negative on a sell; the cost is not."""
    c = PerShareCommission()
    assert c.commission(-500, 100.0) > 0


def test_commission_respects_the_floor():
    c = PerShareCommission(per_share=0.005, minimum=1.0)
    assert c.commission(10, 100.0) == pytest.approx(1.0)      # 10*0.005 = 0.05
    assert c.commission(1000, 100.0) == pytest.approx(5.0)    # above the floor


def test_commission_capped_as_share_of_trade_value():
    """A penny stock must not cost more in commission than a cap allows."""
    c = PerShareCommission(per_share=0.005, minimum=1.0, max_pct_of_value=0.01)
    # 10,000 shares at $0.50 = $5,000 notional. Raw would be $50; cap is $50.
    assert c.commission(10_000, 0.50) == pytest.approx(50.0)
    # At $0.10 the notional is $1,000, so the cap of $10 binds.
    assert c.commission(10_000, 0.10) == pytest.approx(10.0)


def test_zero_commission_is_zero():
    assert ZeroCommission().commission(1000, 100.0) == 0.0


def test_buy_fills_above_mid_and_sell_below():
    s = HalfSpreadSlippage(spread_bps=10.0)
    assert s.fill_price(100, 100.0) > 100.0
    assert s.fill_price(-100, 100.0) < 100.0


def test_half_spread_is_exactly_half_the_full_spread():
    s = HalfSpreadSlippage(spread_bps=10.0)      # 10bp = 0.1% full spread
    # half of 0.1% of 100.0 is 0.05
    assert s.fill_price(1, 100.0) == pytest.approx(100.05)
    assert s.fill_price(-1, 100.0) == pytest.approx(99.95)


def test_zero_slippage_returns_mid():
    assert ZeroSlippage().fill_price(100, 123.45) == 123.45


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        HalfSpreadSlippage(spread_bps=-1)
    with pytest.raises(ValueError):
        PerShareCommission(per_share=-0.01)
    with pytest.raises(ValueError):
        PerShareCommission(max_pct_of_value=0)