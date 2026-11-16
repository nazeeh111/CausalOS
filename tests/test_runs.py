from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from causalos.api import app, _RUNS
from causalos.datasets import example
from causalos import core
from causalos.report import build_report, clean, render_html
from causalos.runs import (AnalysisRun, canonical, create_run, digest, publish_new,
                          read_run, replay_run, save_run, table_snapshot)


@pytest.fixture(scope="module")
def evidence():
    data, metadata = example("confounded", seed=17)
    data = data.head(160)
    question = deepcopy(metadata["question"])
    question.update(seed=19, bootstrap=40)
    return data, question


@pytest.fixture(scope="module")
def run(evidence):
    return create_run(*evidence, include_data=True)


def test_numerical_engine_and_full_refusal_results_unchanged(evidence, run):
    data, question = evidence
    assert run.record["analysis"] == clean(core.analyze(data, question))
    assert run.record["analysis"]["status"] == "identified"
    blocked = deepcopy(question)
    blocked["assumptions"]["exchangeability"] = False
    result = create_run(data, blocked, include_data=True)
    assert result.record["analysis"] == clean(core.analyze(data, blocked))
    assert result.record["analysis"]["status"] == "blocked"
    assert replay_run(result)["status"] == "matched"


def test_snapshot_is_immutable_and_detached_from_callers(evidence, run):
    bundle = run.bundle(include_data=True)
    expected = canonical(bundle)
    loaded = AnalysisRun.open(bundle)
    bundle["record"]["question"]["seed"] = 1
    bundle["data"]["rows"][0][0] = "changed"
    exposed = loaded.record
    exposed["analysis"]["estimates"].clear()
    assert canonical(loaded.bundle(include_data=True)) == expected
    with pytest.raises(Exception):
        loaded._bytes = b"mutated"


def test_run_tamper_and_mismatched_source_rejected(evidence, run):
    for component in ("question", "analysis", "software", "seed"):
        altered = run.bundle(include_data=True)
        altered["record"][component] = {"changed": True} if component != "seed" else 7
        with pytest.raises(ValueError, match="digest"):
            AnalysisRun.open(altered)
    altered = run.bundle(include_data=True)
    altered["data"]["rows"][0][0] = -1234567
    with pytest.raises(ValueError, match="data"):
        AnalysisRun.open(altered)
    data, question = evidence
    changed = data.copy()
    changed.iloc[0, 0] += 1
    with pytest.raises(ValueError, match="Data"):
        build_report(data=changed, run=run)
    with pytest.raises(ValueError, match="Question"):
        build_report(question={**question, "seed": 1}, run=run)
    with pytest.raises(ValueError, match="Independent analysis"):
        build_report(data, question, run.record["analysis"])


def test_replay_refuses_missing_changed_runtime_or_data_and_detects_forgery(evidence, run):
    redacted = AnalysisRun.open(run.bundle())
    assert replay_run(redacted)["status"] == "refused"
    data, _ = evidence
    assert replay_run(redacted, data)["status"] == "matched"
    changed = data.copy()
    changed.iloc[0, 0] += 1
    assert replay_run(redacted, changed)["status"] == "refused"
    with patch("causalos.runs.software_identity", return_value={"different": "runtime"}):
        assert replay_run(run)["status"] == "refused"
    forged = run.bundle(include_data=True)
    forged["record"]["analysis"]["estimates"][0]["estimate"] += 100
    forged["record"]["analysis_sha256"] = digest(forged["record"]["analysis"])
    forged["sha256"] = digest(forged["record"])
    imported = AnalysisRun.open(forged)
    assert replay_run(imported)["status"] == "mismatch"
    assert build_report(run=imported)["analysis_run"]["verification"] == "unverified_saved_result"


def test_save_open_privacy_escaping_and_no_overwrite(tmp_path, run):
    path = tmp_path / "record.json"
    save_run(run, path)
    assert read_run(path).bundle(include_data=True)["data"] is None
    prior = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_run(run, path, include_data=True)
    assert path.read_bytes() == prior
    save_run(run, tmp_path / "private.json", include_data=True)
    assert read_run(tmp_path / "private.json").bundle(include_data=True)["data"] is not None
    report = build_report(run=run)
    assert "data" not in report["analysis_run"]
    assert "rows" not in report["analysis_run"]["record"]
    assert "<script>" not in render_html(build_report(run=run, metadata={"label": "<script>evil</script>"}))
    for failure in ("fsync", "link"):
        target = tmp_path / (failure + ".json")
        with patch("causalos.runs.os." + failure, side_effect=OSError("injected failure")):
            with pytest.raises(OSError):
                save_run(run, target)
        assert not target.exists()
        assert not list(tmp_path.glob(".causalos-*"))
        save_run(run, target)


def test_concurrent_destination_preserved(tmp_path):
    target = tmp_path / "run.json"
    import os
    link = os.link
    def collide(source, destination):
        Path(destination).write_text("other writer")
        return link(source, destination)
    with patch("causalos.runs.os.link", side_effect=collide):
        with pytest.raises(FileExistsError):
            publish_new(target, "our bytes")
    assert target.read_text() == "other writer"
    assert not list(tmp_path.glob(".causalos-*"))


def test_api_save_open_replay_and_untrusted_client_analysis(evidence, run):
    client = TestClient(app)
    data, question = evidence
    records = clean(data.to_dict("records"))
    response = client.post("/api/analyze", json={"data": records, "question": question})
    assert response.status_code == 200
    result = response.json()
    bundle = result["run"]
    assert json.loads(bundle)["data"] is None
    assert result["status"] == "identified"
    report = client.post("/api/report?format=json", json={"run": bundle})
    assert report.json()["analysis_run"]["verification"] == "computed_in_this_process"
    assert client.post("/api/report", json={"analysis": result, "data": records}).status_code == 422
    assert client.post("/api/report", json={"run": bundle, "question": {"seed": 1}}).status_code == 422
    saved = client.post("/api/runs/save", json={"run": bundle, "include_data": True, "data": records}).json()["run"]
    assert json.loads(saved)["data"] is not None
    _RUNS.clear()
    opened = client.post("/api/runs/open", json={"run": saved}).json()
    assert opened["verification"] == "unverified_saved_result"
    assert client.post("/api/report?format=json", json={"run": saved}).json()["analysis_run"]["verification"] == "unverified_saved_result"
    replay = client.post("/api/runs/replay", json={"run": saved}).json()
    assert replay["status"] == "matched"
    assert client.post("/api/report?format=json", json={"run": saved}).json()["analysis_run"]["verification"] == "replayed_in_this_process"
    redacted = client.post("/api/runs/save", json={"run": saved}).json()
    assert json.loads(redacted["run"])["data"] is None
    changed = deepcopy(records)
    changed[0][next(iter(changed[0]))] += 1
    assert client.post("/api/runs/save", json={"run": saved, "include_data": True, "data": changed}).status_code == 422
    assert client.post("/api/runs/replay", json={"run": redacted["run"]}).json()["status"] == "refused"


def test_bounded_file_and_duplicate_key_refusal(tmp_path, run):
    from causalos.runs import MAX_RUN_BYTES
    path = tmp_path / "big.json"
    with path.open("wb") as stream:
        stream.truncate(MAX_RUN_BYTES + 1)
    with pytest.raises(ValueError, match="24 MiB"):
        read_run(path)
    path.write_text('{"record":1,"record":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_run(path)
    assert table_snapshot(pd.DataFrame({"x": [float("nan")]}))["rows"] == [[None]]
    with pytest.raises(ValueError, match="Non-finite"):
        table_snapshot(pd.DataFrame({"x": [float("inf")]}))


def test_loaded_source_and_changed_thread_runtime_refuse_replay(run):
    with patch("causalos.runs._CORE_SOURCE_SHA256", "stale-loaded-source"):
        result = replay_run(run)
        assert result["status"] == "refused"
        assert "restart" in result["reason"]
    with patch("threadpoolctl.threadpool_info", return_value=[{"num_threads": 197}]):
        assert replay_run(run)["status"] == "refused"


def test_close_failure_leaves_no_partial_and_retry_works(tmp_path, run):
    import os
    fdopen = os.fdopen
    class FailClose:
        def __init__(self, *args, **kwargs):
            self.stream = fdopen(*args, **kwargs)
        def __enter__(self):
            return self.stream
        def __exit__(self, *args):
            self.stream.close()
            raise OSError("injected close failure")
    target = tmp_path / "closed.json"
    with patch("causalos.runs.os.fdopen", side_effect=FailClose):
        with pytest.raises(OSError, match="close failure"):
            save_run(run, target)
    assert not target.exists() and not list(tmp_path.glob(".causalos-*"))
    save_run(run, target)


def test_real_javascript_opaque_transport_preserves_number_types_and_replay(run):
    import subprocess
    bundle = run.bundle(include_data=True)
    bundle["record"]["question"]["transport_probe"] = [0.0, -0.0, 1e-20, 1e20, True, 9007199254740993]
    bundle["record"]["question_sha256"] = digest(bundle["record"]["question"])
    bundle["sha256"] = digest(bundle["record"])
    original = canonical(bundle).decode("utf-8")
    javascript = "process.stdout.write(JSON.stringify(JSON.parse(require('fs').readFileSync(0,'utf8'))))"
    response = subprocess.run(["node", "-e", javascript], input=canonical({"run": original}), capture_output=True, check=True)
    transported = json.loads(response.stdout)["run"]
    assert transported == original
    reopened = AnalysisRun.open(transported)
    probes = reopened.record["question"]["transport_probe"]
    assert type(probes[0]) is float and type(probes[-2]) is bool and probes[-1] == 9007199254740993
    assert canonical(reopened.bundle(include_data=True)) == canonical(bundle)
    assert build_report(run=reopened)["analysis"] == run.record["analysis"]
    # Real API response traverses JavaScript, then comes back as opaque run text.
    client = TestClient(app)
    opened = client.post("/api/runs/open", json={"run": canonical(run.bundle(include_data=True)).decode()}).json()
    response = subprocess.run(["node", "-e", javascript], input=canonical(opened), capture_output=True, check=True)
    echoed = json.loads(response.stdout)
    assert client.post("/api/runs/replay", json={"run": echoed["run"]}).json()["status"] == "matched"


def test_run_endpoints_reject_nested_cells_before_execution(run):
    client = TestClient(app)
    for path in ("/api/runs/save", "/api/runs/replay", "/api/report"):
        response = client.post(path, json={"run": canonical(run.bundle()).decode(), "data": [{"nested": {"value": 1}}]})
        assert response.status_code == 422


def test_exact_scalar_api_identity_and_lossy_integer_refusal():
    client = TestClient(app, raise_server_exceptions=False)
    question = {"assumptions": {"annotation": 9007199254740993}}
    rows = [{"X": 1, "note": "raw-sentinel"}, {"X": None, "note": "b"}]
    response = client.post("/api/analyze", json={"data": rows, "question": question, "include_data": True})
    assert response.status_code == 200
    run = json.loads(response.json()["run"])
    assert type(run["data"]["rows"][0][0]) is int
    changed = deepcopy(rows)
    changed[0]["X"] = True
    assert client.post("/api/runs/replay", json={"run": run, "data": changed}).json()["status"] == "refused"
    assert client.post("/api/report", json={"run": run, "data": changed}).status_code == 422
    rows[0]["X"] = 9007199254740993
    refused = client.post("/api/analyze", json={"data": rows, "question": question})
    assert refused.status_code == 422 and "exactly" in refused.text


def test_discrete_analysis_and_software_identity_cannot_acquire_replay_trust():
    client = TestClient(app, raise_server_exceptions=False)
    question = {"assumptions": {"annotation": 9007199254740993, "fractional_note": 0.5}}
    run = create_run([{"X": 1}, {"X": None}], question, include_data=True)
    for replacement in (9007199254740994, 10 ** 350):
        forged = run.bundle(include_data=True)
        forged["record"]["analysis"]["assumptions"]["annotation"] = replacement
        forged["record"]["analysis_sha256"] = digest(forged["record"]["analysis"])
        forged["sha256"] = digest(forged["record"])
        replay = client.post("/api/runs/replay", json={"run": forged})
        assert replay.status_code == 200 and replay.json()["status"] == "mismatch"
        report = client.post("/api/report?format=json", json={"run": forged}).json()
        assert report["analysis_run"]["verification"] == "unverified_saved_result"
    forged = run.bundle(include_data=True)
    forged["record"]["analysis"]["assumptions"]["fractional_note"] += 1e-12
    forged["record"]["analysis_sha256"] = digest(forged["record"]["analysis"])
    forged["sha256"] = digest(forged["record"])
    assert replay_run(AnalysisRun.open(forged))["status"] == "mismatch"
    forged = run.bundle(include_data=True)
    forged["record"]["software"]["numerical_libraries"][0]["num_threads"] = True
    forged["sha256"] = digest(forged["record"])
    assert client.post("/api/runs/replay", json={"run": forged}).json()["status"] == "refused"


def test_replay_uses_one_detached_snapshot_during_real_concurrent_mutation():
    from threading import Event, Thread
    from causalos import runs
    data = pd.DataFrame({"X": [1.0, 2.0], "T": [0, 1], "Y": [3.0, 4.0]})
    run = create_run(data, {})
    snap_done, altered = Event(), Event()
    original_snapshot, original_analyze = runs.table_snapshot, runs.core.analyze
    seen = []
    def writer():
        assert snap_done.wait(3)
        data.loc[0, "X"] = 123.0
        altered.set()
    def snapshot(source):
        table = original_snapshot(source)
        if source is data:
            snap_done.set()
            assert altered.wait(3)
        return table
    def capture(frame, question):
        seen.append(frame.loc[0, "X"])
        return original_analyze(frame, question)
    thread = Thread(target=writer)
    thread.start()
    try:
        with patch.object(runs, "table_snapshot", snapshot), patch.object(runs.core, "analyze", capture):
            result = replay_run(run, data)
    finally:
        thread.join(3)
    assert seen == [1.0] and result["status"] == "matched"
    assert data.loc[0, "X"] == 123.0


def test_http_wire_bound_precedes_json_admission_for_declared_and_streamed_bodies():
    import asyncio
    from causalos.api import BoundedRequestBody
    async def attempt(headers, chunks):
        called, messages = [], []
        async def downstream(scope, receive, send):
            called.append((await receive())["body"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})
        pending = [{"type": "http.request", "body": chunk, "more_body": i + 1 < len(chunks)} for i, chunk in enumerate(chunks)]
        async def receive():
            return pending.pop(0)
        async def send(message):
            messages.append(message)
        scope = {"type": "http", "method": "POST", "path": "/api/runs/open", "headers": headers}
        with patch("causalos.api.MAX_HTTP_BYTES", 128):
            await BoundedRequestBody(downstream)(scope, receive, send)
        return called, messages[0]["status"]
    for headers, chunks in (([(b"content-length", b"129")], [b"x"]),
                            ([], [b"x" * 64, b"x" * 65]),
                            ([(b"content-length", b"1")], [b"x" * 129])):
        called, status = asyncio.run(attempt(headers, chunks))
        assert not called and status == 413
    called, status = asyncio.run(attempt([], [b"x" * 64, b"x" * 64]))
    assert called == [b"x" * 128] and status == 200
    assert asyncio.run(attempt([(b"content-length", b"invalid")], [b"x"]))[1] == 400


def test_extra_envelope_fields_refused_and_csv_preserves_then_refuses_large_integer(tmp_path, run):
    from causalos.runs import read_csv_table
    client = TestClient(app)
    assert client.post("/api/runs/open", json={"run": canonical(run.bundle()).decode(), "padding": "ignored before"}).status_code == 422
    source = tmp_path / "exact.csv"
    source.write_text("X,T\n1,True\n,False\n")
    table = read_csv_table(source)
    assert table == [{"X": 1, "T": True}, {"X": None, "T": False}]
    assert type(table[0]["X"]) is int
    source.write_text("X\n9007199254740993\n\n")
    with pytest.raises(ValueError, match="exactly"):
        read_csv_table(source)


def test_tolerance_only_for_measured_floating_outputs_not_configuration(run):
    from causalos.runs import analysis_matches
    original = run.record["analysis"]
    changed = deepcopy(original)
    changed["estimates"][0]["estimate"] += 1e-12
    assert analysis_matches(original, changed)
    changed = deepcopy(original)
    changed["estimates"][0]["confidence_level"] += 1e-12
    assert not analysis_matches(original, changed)
    changed = deepcopy(original)
    changed["sensitivity"]["grid"][0]["r2_treatment"] += 1e-12
    assert not analysis_matches(original, changed)
    changed = deepcopy(original)
    changed["diagnostics"]["n"] += 1
    assert not analysis_matches(original, changed)
