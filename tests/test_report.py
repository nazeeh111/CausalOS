import json

import numpy as np
import pandas as pd

from causalos.report import build_report, clean, render_html
from causalos.runs import create_run


def test_report_escapes_user_content_and_retains_assumptions():
    table = pd.DataFrame({'<script>alert(1)</script>': [1, 2]})
    question = {'assumptions': {'exchangeability': False}, 'outcome': '<img src=x onerror=alert(1)>'}
    run = create_run(table, question)
    report = build_report(table, question, run=run)
    html = render_html(report)
    assert '<script>alert' not in html
    assert '<img src=x' not in html
    assert 'exchangeability' in html
    assert 'blocked' in html
    assert report['dataset']['sha256'] == run.record['dataset']['sha256']
    json.dumps(report, allow_nan=False)


def test_json_conversion_never_emits_nan():
    assert clean({'x': np.nan, 'y': np.int64(2), 'z': [float('inf')]}) == {'x': None, 'y': 2, 'z': [None]}
