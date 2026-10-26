import pytest

from causalos.counterfactual import compare
from causalos.design import initial_state


def test_same_unit_consistency_and_competing_predictions():
    state = initial_state()
    observed = {'A': .4, 'Y': -.2, 'B': 0., 'Z': .5}
    same = compare(state, observed, 'A', .4)
    assert all(p['predicted'] == observed for p in same['models'])
    changed = compare(state, observed, 'A', 1.)
    assert changed['models'][0]['change_in_Y'] == pytest.approx(state['rho'] * .6)
    assert changed['models'][2]['change_in_Y'] == 0
    assert observed['A'] == .4


def test_invalid_counterfactual_inputs_are_rejected():
    with pytest.raises(ValueError):
        compare(initial_state(), {'A': 0}, 'A', 1)
