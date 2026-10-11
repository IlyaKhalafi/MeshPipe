import pytest

from meshpipe.adaptive import AdaptiveBudget


def test_budget_hysteresis_and_bounds():
    control = AdaptiveBudget(640, 30)
    control.configure(True, 33)
    for _ in range(29):
        control.observe(80)
    assert control.width == 640
    control.observe(80)
    assert control.width < 640 and control.fps < 30
    for _ in range(600):
        control.observe(80)
    assert control.width == 160
    for _ in range(900):
        control.observe(5)
    assert control.width == 640 and control.fps == 30
    control.configure(False, 33)
    assert control.width == 640


@pytest.mark.parametrize("budget", [0, float("nan"), 201])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        AdaptiveBudget(640, 30, budget)
