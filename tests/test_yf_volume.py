"""The yfinance dollar-volume repair.

Same conservation law as the CRSP adapter, applied to a file that already
exists rather than to one being loaded. Kept in its own module because the
repair is a one-time script, not part of the engine.
"""
import numpy as np
import pandas as pd
import pytest

from fix_yfinance_volume import corrected


def make_pair(n=60, ret=0.0005, div=0.0002, price=200.0, volume=1_000_000):
    """An adjusted frame and its raw close, the way yfinance serves them.

    `close` compounds the TOTAL return. `raw_close` compounds the return net
    of dividends, and is anchored so the two agree on the LAST bar -- which
    is how yfinance's back-adjustment works.
    """
    dates = pd.bdate_range("2015-01-02", periods=n)
    close = price * (1 + ret) ** np.arange(n)
    raw = price * (1 + ret - div) ** np.arange(n)
    raw = raw * (close[-1] / raw[-1])            # anchor at the last bar
    adjusted = pd.DataFrame({
        "open": close * 0.995, "high": close * 1.005,
        "low": close * 0.990, "close": close,
        "volume": np.full(n, float(volume)),
    }, index=dates)
    return adjusted, pd.Series(raw, index=dates)


def test_dollar_volume_is_preserved():
    """close * volume == raw close * raw volume, on every bar."""
    adjusted, raw = make_pair()
    out = corrected(adjusted, raw)
    got = out["close"].to_numpy() * out["volume"].to_numpy()
    want = raw.to_numpy() * adjusted["volume"].to_numpy()
    assert got == pytest.approx(want)


def test_prices_are_not_touched():
    """The prices were never wrong. Only the basis mismatch was."""
    adjusted, raw = make_pair()
    out = corrected(adjusted, raw)
    for col in ("open", "high", "low", "close"):
        assert out[col].to_numpy() == pytest.approx(adjusted[col].to_numpy())


def test_returns_are_unchanged():
    """A repair that moves the return series has broken the thing it fixed."""
    adjusted, raw = make_pair()
    out = corrected(adjusted, raw)
    assert out["close"].pct_change().dropna().to_numpy() == pytest.approx(
        adjusted["close"].pct_change().dropna().to_numpy())


def test_early_volume_is_revised_upward():
    """Early adjusted prices are below raw, so the same dollars are more
    synthetic shares. Understating them is what shrank the capacity wall."""
    adjusted, raw = make_pair()
    out = corrected(adjusted, raw)
    assert out["volume"].iloc[0] > adjusted["volume"].iloc[0]
    assert out["volume"].iloc[-1] == pytest.approx(adjusted["volume"].iloc[-1])


def test_a_frame_already_on_one_basis_is_unchanged():
    """No dividends means no mismatch, and the repair must be a no-op.
    A 'fix' that changes correct data is not a fix."""
    adjusted, _ = make_pair(div=0.0)
    out = corrected(adjusted, adjusted["close"])
    assert out["volume"].to_numpy() == pytest.approx(adjusted["volume"].to_numpy())


def test_missing_raw_close_is_rejected():
    adjusted, raw = make_pair()
    with pytest.raises(ValueError, match="no raw close"):
        corrected(adjusted, raw.iloc[:-5])


def test_non_positive_raw_close_is_rejected():
    adjusted, raw = make_pair()
    raw.iloc[3] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        corrected(adjusted, raw)


def test_the_repair_lowers_measured_participation():
    """The payoff, asserted end to end rather than argued.

    Same strategy, same capital, same cost models. The only change is a
    volume column on the right basis, and impact falls because participation
    was overstated.
    """
    import queue

    from backtester.costs import SquareRootImpact
    from backtester.data import HistoricBarHandler
    from backtester.events import OrderEvent
    from backtester.execution import SimulatedExecutionHandler

    adjusted, raw = make_pair()
    out = corrected(adjusted, raw)

    def impact_on(frame):
        data = HistoricBarHandler(frame, "SPY")
        events: queue.Queue = queue.Queue()
        for _ in range(30):
            data.update_bars(events)
        while not events.empty():
            events.get()
        handler = SimulatedExecutionHandler(impact=SquareRootImpact(1.0))
        handler.on_order(OrderEvent(data.current_time, "SPY", 500_000),
                         events, data)
        return events.get().impact_cost

    assert impact_on(out) < impact_on(adjusted)


def test_capacity_headroom_moves_out_by_the_factor():
    """The participation cap is 10% of ADV. Correcting ADV upward moves the
    wall out by exactly the same ratio, which is where ~$2.7bn -> ~$3.3bn
    comes from."""
    adjusted, raw = make_pair()
    out = corrected(adjusted, raw)

    window = slice(0, 21)
    before = adjusted["volume"].iloc[window].mean()
    after = out["volume"].iloc[window].mean()
    factor = (raw / adjusted["close"]).iloc[window].mean()
    assert after / before == pytest.approx(factor, rel=1e-6)
    assert factor > 1.0