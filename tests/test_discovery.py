import numpy as np
import pandas as pd
import pytest

from causalos.discovery import discover


def test_observational_pair_remains_unoriented():
    pytest.importorskip('causallearn')
    rng = np.random.default_rng(41)
    x = rng.normal(size=1000)
    result = discover(pd.DataFrame({'X': x, 'Y': x + rng.normal(size=1000)}))
    assert result['identifiable'] is False
    assert result['graph_source'] == 'learned'
    assert result['graph']['unresolved'] == [['X', 'Y']]
    assert result['graph']['edges'] == []


def test_discrete_or_missing_data_rejected():
    pytest.importorskip('causallearn')
    with pytest.raises(ValueError, match='continuous'):
        discover(pd.DataFrame({'A': [0, 1] * 30, 'B': np.arange(60)}))
    with pytest.raises(ValueError, match='missing'):
        discover(pd.DataFrame({'A': [np.nan] + list(range(59)), 'B': np.arange(60)}))
