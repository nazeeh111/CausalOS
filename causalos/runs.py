"""Immutable, content-bound analysis records. Hash integrity is not authentication."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import copy
import csv
import io
import re
import importlib.metadata
import importlib.util
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import pandas as pd

from . import __version__, core
from .report import clean

MAX_RUN_BYTES = 24 * 1024 * 1024
RTOL, ATOL = 1e-9, 1e-12
_CORE_SOURCE_SHA256 = hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("Run must contain finite, bounded JSON values") from exc


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


@lru_cache(maxsize=1)
def runtime_builds():
    # Binary hashes distinguish numerical builds that share a package version.
    binaries = {}
    for name, candidates in (("numpy_multiarray", ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath")),
                             ("scipy_blas", ("scipy.linalg._fblas",)), ("sklearn_tree", ("sklearn.tree._tree",))):
        for module in candidates:
            try:
                spec = importlib.util.find_spec(module)
            except ModuleNotFoundError:
                continue
            if spec is not None and spec.origin:
                binaries[name] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
                break
    return binaries


def software_identity():
    from threadpoolctl import threadpool_info
    current = hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    if current != _CORE_SOURCE_SHA256:
        raise ValueError("Numerical source changed after this process loaded it; restart before analysis or replay")
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "pandas", "scikit-learn", "statsmodels", "networkx", "threadpoolctl")}
    return {"causalos": __version__, "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages, "binary_sha256": runtime_builds(),
            "numerical_libraries": [{k: v for k, v in item.items() if k != "filepath"} for item in threadpool_info()],
            "core_sha256": current}


def validate_table(table):
    if type(table) is not dict or set(table) != {"columns", "rows"}:
        raise ValueError("Invalid scalar table")
    columns, rows = table["columns"], table["rows"]
    if type(columns) is not list or type(rows) is not list or not 1 <= len(rows) <= 20000 or not 1 <= len(columns) <= 40:
        raise ValueError("Use 1–20000 rows and 1–40 columns")
    if any(type(c) is not str or not c or len(c) > 100 for c in columns) or len(set(columns)) != len(columns):
        raise ValueError("Columns must be unique nonempty names up to 100 characters")
    for row in rows:
        if type(row) is not list or len(row) != len(columns):
            raise ValueError("Table rows disagree with columns")
        for cell in row:
            if cell is not None and type(cell) not in (str, int, float, bool):
                raise ValueError("Data must contain only scalar text, numbers, booleans or null")
            if type(cell) is int:
                try:
                    exact = int(float(cell)) == cell
                except (OverflowError, ValueError):
                    exact = False
                if not exact:
                    raise ValueError("An integer cannot be represented exactly by this float64 numerical engine; use an audited transformation first")
            if type(cell) is float and not math.isfinite(cell):
                raise ValueError("Non-finite scalar data is unsupported; provide an explicit missing-data representation")
            if isinstance(cell, str) and len(cell) > 1000:
                raise ValueError("Text cells must be at most 1000 characters")
    return table


def table_snapshot(data):
    if isinstance(data, pd.DataFrame):
        detached = data.copy(deep=True)
        rows = detached.astype(object).where(pd.notna(detached), None).values.tolist()
        for row in rows:
            for cell in row:
                if cell is not None and clean(cell) is None:
                    raise ValueError("Non-finite scalar data is unsupported; provide an explicit missing-data representation")
        table = {"columns": list(detached.columns), "rows": clean(rows)}
    elif type(data) is list:
        detached = copy.deepcopy(data)
        if not detached or any(type(row) is not dict for row in detached):
            raise ValueError("Supply a nonempty list of scalar records")
        columns = list(dict.fromkeys(key for row in detached for key in row))
        table = {"columns": columns, "rows": [[row.get(key) for key in columns] for row in detached]}
    else:
        raise ValueError("Supply a dataframe or scalar records")
    return validate_table(table)


def table_frame(table):
    validate_table(table)
    # Inference is permitted only after the original scalar values were bound.
    # Check value preservation explicitly; int/bool identity stays in the table.
    frame = pd.DataFrame(table["rows"], columns=table["columns"])
    actual = clean(frame.astype(object).where(pd.notna(frame), None).values.tolist())
    for original, converted in zip(table["rows"], actual):
        for a, b in zip(original, converted):
            if type(a) is bool or type(b) is bool:
                equal = type(a) is type(b) and a == b
            elif type(a) in (int, float) and type(b) in (int, float):
                equal = a == b
            else:
                equal = type(a) is type(b) and a == b
            if not equal:
                raise ValueError("DataFrame conversion would change a bound scalar value")
    return frame


def read_csv_table(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_RUN_BYTES + 1)
    if len(raw) > MAX_RUN_BYTES:
        raise ValueError("CSV exceeds the 24 MiB input limit")
    parsed = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    columns = next(parsed, [])
    rows = []
    for values in parsed:
        if not values:
            continue
        row = []
        for value in values:
            if value == "":
                row.append(None)
            elif re.fullmatch(r"[+-]?[0-9]+", value):
                row.append(int(value))
            elif re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value):
                row.append(float(value))
            elif value in ("True", "False"):
                row.append(value == "True")
            else:
                row.append(value)
        rows.append(row)
        if len(rows) > 20000:
            raise ValueError("Use at most20000 CSV rows")
    table = validate_table({"columns": columns, "rows": rows})
    return [dict(zip(columns, row)) for row in table["rows"]]


def dataset_identity(table):
    return {"rows": len(table["rows"]), "columns": table["columns"], "sha256": digest(table)}


def validate_bundle(bundle):
    if type(bundle) is not dict or set(bundle) != {"record", "sha256", "data"}:
        raise ValueError("Expected a versioned analysis-run bundle")
    encoded = canonical(bundle)
    if len(encoded) > MAX_RUN_BYTES:
        raise ValueError("Run exceeds the 24 MiB limit")
    record = bundle["record"]
    keys = {"format", "version", "created_utc", "dataset", "question", "question_sha256", "seed", "software", "analysis", "analysis_sha256"}
    if type(record) is not dict or set(record) != keys or record["format"] != "causalos-analysis-run" or type(record["version"]) is not int or record["version"] != 1:
        raise ValueError("Unsupported analysis-run schema/version")
    if bundle["sha256"] != digest(record):
        raise ValueError("Run content digest mismatch")
    if type(record["question"]) is not dict or record["question_sha256"] != digest(record["question"]) or canonical(record["seed"]) != canonical(record["question"].get("seed", 42)):
        raise ValueError("Question/seed binding mismatch")
    if type(record["analysis"]) is not dict or record["analysis_sha256"] != digest(record["analysis"]) or record["analysis"].get("status") not in ("identified", "blocked"):
        raise ValueError("Analysis binding/status mismatch")
    info = record["dataset"]
    if type(info) is not dict or set(info) != {"rows", "columns", "sha256"} or type(info["rows"]) is not int or not 1 <= info["rows"] <= 20000 or type(info["columns"]) is not list or not 1 <= len(info["columns"]) <= 40:
        raise ValueError("Invalid dataset identity")
    if len(set(info["columns"])) != len(info["columns"]) or any(type(c) is not str or not c or len(c) > 100 for c in info["columns"]):
        raise ValueError("Invalid dataset columns")
    if type(info["sha256"]) is not str or len(info["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in info["sha256"]):
        raise ValueError("Invalid dataset digest")
    if type(record["software"]) is not dict or type(record["created_utc"]) is not str:
        raise ValueError("Invalid software/time identity")
    if bundle["data"] is not None:
        table_frame(bundle["data"])
        if dataset_identity(bundle["data"]) != info:
            raise ValueError("Embedded data digest mismatch")
    return encoded


@dataclass(frozen=True)
class AnalysisRun:
    _bytes: bytes

    @classmethod
    def open(cls, bundle):
        if isinstance(bundle, (str, bytes)):
            bundle = parse_run_json(bundle)
        return cls(validate_bundle(bundle))

    def bundle(self, *, include_data=False):
        result = json.loads(self._bytes)
        if not include_data:
            result["data"] = None
        return result

    @property
    def record(self):
        return self.bundle()["record"]

    @property
    def id(self):
        return self.bundle()["sha256"]

    def with_data(self, data):
        table = table_snapshot(data)
        bundle = self.bundle()
        if dataset_identity(table) != bundle["record"]["dataset"]:
            raise ValueError("Data does not match the saved run")
        bundle["data"] = table
        return self.open(bundle)

    def check_inputs(self, data=None, question=None):
        record = self.record
        if data is not None and dataset_identity(table_snapshot(data)) != record["dataset"]:
            raise ValueError("Data does not match the analysis run")
        if question is not None and digest(clean(question)) != record["question_sha256"]:
            raise ValueError("Question does not match the analysis run")


def create_run(data, question, *, include_data=False):
    # Snapshot first, then analyze that same representation; caller mutation
    # cannot reassign a result to different input bytes after computation.
    table = table_snapshot(data)
    question = json.loads(canonical(clean(question)))
    software = software_identity()
    analysis = clean(core.analyze(table_frame(table), question))
    if canonical(software) != canonical(software_identity()):
        raise ValueError("Runtime changed during analysis; rerun in a stable process")
    record = {"format": "causalos-analysis-run", "version": 1,
              "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": dataset_identity(table),
              "question": question, "question_sha256": digest(question), "seed": question.get("seed", 42),
              "software": software, "analysis": analysis, "analysis_sha256": digest(analysis)}
    return AnalysisRun.open({"record": record, "sha256": digest(record), "data": table if include_data else None})


FLOAT_MEASUREMENTS = {"estimate", "se", "ci", "min", "max", "outside_005_095", "clipped_fraction",
                      "effective_sample_size_treated", "effective_sample_size_control", "smd_before", "smd_after_ipw",
                      "bias", "adjusted_effect", "effect_lower", "effect_upper"}


def close_results(a, b, *, floating=False, field=None):
    if type(a) is not type(b):
        return False
    if type(a) is float:
        return math.isfinite(a) and math.isfinite(b) and (math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL) if floating and field in FLOAT_MEASUREMENTS else a == b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close_results(a[k], b[k], floating=floating, field=k) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close_results(x, y, floating=floating, field=field) for x, y in zip(a, b))
    return a == b


def analysis_matches(saved, computed):
    measured = {"estimates", "diagnostics", "refutation", "sensitivity"}
    return saved.keys() == computed.keys() and all(close_results(saved[key], computed[key], floating=key in measured) for key in saved)


def replay_run(run, data=None):
    record = run.record
    try:
        current_software = software_identity()
    except ValueError as exc:
        return {"status": "refused", "reason": str(exc)}
    if canonical(record["software"]) != canonical(current_software):
        return {"status": "refused", "reason": "Numerical source or runtime identity changed; inspect the saved record without replay."}
    if data is None:
        table = run.bundle(include_data=True)["data"]
        if table is None:
            return {"status": "refused", "reason": "Raw data was omitted. Supply the matching dataset to replay."}
    else:
        try:
            table = table_snapshot(data)
        except ValueError as exc:
            return {"status": "refused", "reason": str(exc)}
    if canonical(dataset_identity(table)) != canonical(record["dataset"]):
        return {"status": "refused", "reason": "Data does not match the saved run"}
    # Identity and computation consume this one detached table, never the caller
    # again. A concurrent mutation after snapshot cannot change the engine input.
    result = clean(core.analyze(table_frame(table), record["question"]))
    match = analysis_matches(record["analysis"], result)
    return {"status": "matched" if match else "mismatch", "run_sha256": run.id,
            "analysis_status": result["status"], "rtol": RTOL, "atol": ATOL,
            "reason": "All numerical and refusal fields reproduced within tolerance." if match else "Recomputed results differ; the saved record was not changed."}


def parse_run_json(raw):
    size = len(raw.encode("utf-8")) if isinstance(raw, str) else len(raw)
    if size > MAX_RUN_BYTES:
        raise ValueError("Run exceeds the 24 MiB limit")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate run JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("Non-finite JSON value")
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("Invalid bounded run JSON") from exc


def read_run(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_RUN_BYTES + 1)
    return AnalysisRun.open(raw)


def publish_new(path, text):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".causalos-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def save_run(run, path, *, include_data=False):
    publish_new(path, canonical(run.bundle(include_data=include_data)).decode("utf-8"))
