import math

import numpy as np
import pandas as pd
import pytest

from backtester.costs import SquareRootImpact, ZeroImpact


def test_zero_impact_is_zero():
    assert ZeroImpact().price_move(1_000, 100.0, 1e6, 0.01) == 0.0


def test_zero_quantity_has_no_impact():
    assert SquareRootImpact().price_move(0, 100.0, 1e6, 0.01) == 0.0


def test_full_adv_costs_one_sigma():
    """The anchor the whole model hangs on."""
    m = SquareRootImpact(coefficient=1.0)
    move = m.price_move(1_000_000, 100.0, 1_000_000, 0.012)
    assert move == pytest.approx(100.0 * 0.012)


def test_impact_per_share_scales_as_square_root():
    m = SquareRootImpact()
    small = m.price_move(10_000, 100.0, 1e7, 0.01)
    large = m.price_move(20_000, 100.0, 1e7, 0.01)
    assert large / small == pytest.approx(math.sqrt(2))


def test_total_impact_cost_scales_as_three_halves():
    """Per share it is sqrt(Q). In dollars it is Q^1.5. This is the one
    people get wrong, and it is why capacity is a wall not a slope."""
    m = SquareRootImpact()
    small = 10_000 * m.price_move(10_000, 100.0, 1e7, 0.01)
    large = 20_000 * m.price_move(20_000, 100.0, 1e7, 0.01)
    assert large / small == pytest.approx(2 ** 1.5)


def test_impact_is_positive_for_sells():
    m = SquareRootImpact()
    assert m.price_move(-50_000, 100.0, 1e7, 0.01) > 0


def test_sells_and_buys_have_symmetric_magnitude():
    m = SquareRootImpact()
    assert m.price_move(-50_000, 100.0, 1e7, 0.01) == pytest.approx(
        m.price_move(50_000, 100.0, 1e7, 0.01))


def test_nan_adv_returns_zero_not_nan():
    """NaN fails every comparison. If this returns NaN the whole equity
    curve becomes NaN silently."""
    m = SquareRootImpact()
    assert m.price_move(1_000, 100.0, float("nan"), 0.01) == 0.0
    assert m.price_move(1_000, 100.0, 1e6, float("nan")) == 0.0


def test_negative_coefficient_is_rejected():
    with pytest.raises(ValueError):
        SquareRootImpact(coefficient=-1.0)