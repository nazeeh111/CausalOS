"""Abduction-action-prediction inside the demo's explicit, fully observed SCMs."""
from __future__ import annotations

import math

from causalos.design import _validate_state


def compare(state: dict, observed: dict[str, float], target: str, value: float) -> dict:
    """Hold each model's inferred exogenous disturbances fixed under intervention.

    These predictions are identified within a *specified* invertible SCM, not
    across the competing models or for people represented by an uploaded CSV.
    """
    _validate_state(state)
    names = ['A', 'Y', 'B', 'Z']
    if target not in names or set(observed) != set(names):
        raise ValueError('Provide observed A,Y,B,Z and a target among those variables.')
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in [*observed.values(), value]):
        raise ValueError('Observed values and intervention must be finite numbers.')
    rho = state['rho']
    predictions = []
    for i, model in enumerate(state['hypotheses']):
        result = dict(observed)
        result[target] = float(value)
        for source, child in model['edges']:
            if source == target:
                coefficient = rho if {source, child} == {'A', 'Y'} else .8
                result[child] += coefficient * (value - observed[source])
        predictions.append({'hypothesis': model['id'], 'label': model['label'],
                            'probability': state['posterior'][i], 'predicted': result,
                            'change_in_Y': result['Y'] - observed['Y']})
    return {'evidence_status': 'structural-model-conditional counterfactual',
            'observed': observed, 'intervention': {'target': target, 'value': value},
            'models': predictions,
            'posterior_mean_Y': sum(p['probability'] * p['predicted']['Y'] for p in predictions),
            'limitations': [
                'Hypothetical unit within the stated linear Gaussian SCMs. No real participant outcome is predicted.',
                'Each model infers exogenous noise from all four observed variables, then holds that noise fixed.',
                'Competing graphs imply different counterfactuals. Their posterior average does not identify an individual causal outcome.',
                'This comparison is unavailable for uploaded datasets without a fully specified structural model.',
            ]}
