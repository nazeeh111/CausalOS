import json
import sys

from causalos import cli


def test_benchmark_cli_does_not_overwrite_output_directory(tmp_path, monkeypatch):
    def tiny_benchmark(output):
        output.mkdir()
        (output / 'summary.json').write_text('{"status":"test"}')
    monkeypatch.setattr('causalos.benchmark.run_benchmark', tiny_benchmark)
    monkeypatch.setattr(sys, 'argv', ['causalos', 'benchmark', '--out', str(tmp_path / 'benchmark')])
    cli.main()
    assert json.loads((tmp_path / 'benchmark' / 'summary.json').read_text())['status'] == 'test'
