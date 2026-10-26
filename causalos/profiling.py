"""DuckDB-backed, read-only profiling with reproducible input provenance."""
from __future__ import annotations
import hashlib
import json
import math
from typing import Any
import duckdb
import numpy as np
import pandas as pd


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _clean(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def profile(data: pd.DataFrame, treatment: str | None = None, covariates: list[str] | None = None) -> dict:
    if not isinstance(data, pd.DataFrame) or len(data.columns) == 0:
        raise ValueError("Supply a table with at least one column.")
    if data.columns.duplicated().any() or not all(isinstance(c, str) for c in data.columns):
        raise ValueError("Column names must be unique strings.")
    if len(data) > 200000 or len(data.columns) > 200:
        raise ValueError("Interactive profiling supports up to 200,000 rows and 200 columns.")
    db = duckdb.connect(":memory:")
    queries = []
    try:
        db.register("observations", data.copy(deep=True))
        query = "SELECT COUNT(*) AS rows FROM observations"
        queries.append(query)
        n = int(db.execute(query).fetchone()[0])
        query = "SELECT COUNT(*) FROM (SELECT DISTINCT * FROM observations)"
        queries.append(query)
        duplicates = n - int(db.execute(query).fetchone()[0])
        columns = []
        for c in data:
            qc = _quote(c)
            query = f"SELECT COUNT(*) FILTER (WHERE {qc} IS NULL), COUNT(DISTINCT {qc}) FROM observations"
            queries.append(query)
            missing, unique = db.execute(query).fetchone()
            col = {"name": c, "dtype": str(data[c].dtype), "missing": int(missing), "missing_percent": float(missing / n * 100) if n else 0, "unique": int(unique), "numeric": None}
            if pd.api.types.is_numeric_dtype(data[c]) and not pd.api.types.is_bool_dtype(data[c]):
                query = f"SELECT AVG({qc}), STDDEV_SAMP({qc}), MIN({qc}), QUANTILE_CONT({qc}, 0.25), MEDIAN({qc}), QUANTILE_CONT({qc}, 0.75), MAX({qc}), COUNT(*) FILTER(WHERE NOT ISFINITE({qc})) FROM observations"
                queries.append(query)
                values = db.execute(query).fetchone()
                col["numeric"] = dict(zip(["mean", "std", "min", "q25", "median", "q75", "max"], [_clean(v) for v in values[:7]]))
                col["nonfinite"] = int(values[7])
            columns.append(col)
        numeric = [c for c in data if pd.api.types.is_numeric_dtype(data[c]) and data[c].nunique() > 1][:24]
        corr = data[numeric].corr().to_numpy() if numeric else np.empty((0, 0))
        cohorts = []
        balance = []
        if treatment is not None:
            if treatment not in data:
                raise ValueError("Treatment column is absent.")
            qt = _quote(treatment)
            query = f"SELECT {qt} AS treatment, COUNT(*) AS n FROM observations GROUP BY {qt} ORDER BY {qt} LIMIT 30"
            queries.append(query)
            cohorts = [{"treatment": _clean(v), "n": int(count)} for v, count in db.execute(query).fetchall()]
            if set(data[treatment].dropna().unique()) == {0, 1}:
                for c in covariates or []:
                    if c not in data or not pd.api.types.is_numeric_dtype(data[c]):
                        continue
                    qc = _quote(c)
                    query = f"SELECT {qt}, AVG({qc}), VAR_SAMP({qc}) FROM observations WHERE {qt} IN (0,1) GROUP BY {qt} ORDER BY {qt}"
                    queries.append(query)
                    groups = db.execute(query).fetchall()
                    if len(groups) == 2 and all(g[1] is not None and g[2] is not None for g in groups):
                        sd = math.sqrt((groups[0][2] + groups[1][2]) / 2)
                        balance.append({"variable": c, "smd_before": float((groups[1][1] - groups[0][1]) / sd) if sd else 0.0})
        schema = json.dumps([(c, str(data[c].dtype)) for c in data], separators=(",", ":"))
        digest = hashlib.sha256(schema.encode() + pd.util.hash_pandas_object(data, index=False).to_numpy().tobytes()).hexdigest()
        warnings = []
        if duplicates:
            warnings.append(f"{duplicates} repeated row records detected; no rows removed. Repeated observations may require clustered inference.")
        if any(c["missing"] for c in columns):
            warnings.append("Missing values detected. Choose and document a missing-data plan before estimation.")
        if any(c.get("nonfinite", 0) for c in columns):
            warnings.append("Non-finite numeric values detected; estimation will block these inputs.")
        return {"rows": n, "columns": columns, "duplicates": duplicates, "correlation": {"columns": numeric, "values": [[_clean(v) for v in row] for row in corr]}, "cohorts": cohorts, "balance": balance, "provenance": {"sha256": digest, "hash_definition": "SHA256 of ordered (column,dtype) JSON and pandas hash_pandas_object(index=False) bytes; pandas-version dependent", "engine": "DuckDB " + duckdb.__version__, "transformations": [], "queries": queries, "rows_removed": 0}, "warnings": warnings}
    finally:
        db.close()


def audited_transform(data: pd.DataFrame, operations: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Apply a limited explicit SQL transformation plan and retain every decision.

    Supported operations: drop_missing(columns), standardize(columns), and
    rename(mapping). No arbitrary SQL is accepted. Rows are never removed unless
    a drop_missing step explicitly names its columns.
    """
    if not isinstance(operations, list) or len(operations) > 20:
        raise ValueError("Supply at most 20 explicit transformation operations.")
    original = profile(data)["provenance"]["sha256"]
    current = data.copy(deep=True)
    steps = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("Each operation must be an object.")
        kind = operation.get("op")
        before = len(current)
        db = duckdb.connect(":memory:")
        try:
            db.register("observations", current.copy(deep=True))
            if kind in {"drop_missing", "standardize"}:
                columns = operation.get("columns", [])
                if not isinstance(columns, list) or not columns or any(c not in current for c in columns) or len(columns) != len(set(columns)):
                    raise ValueError("Choose unique existing columns for the operation.")
            if kind == "drop_missing":
                query = "SELECT * FROM observations WHERE " + " AND ".join(_quote(c) + " IS NOT NULL" for c in columns)
                current = db.execute(query).df()
                details = {"warning": "Complete-case selection can induce selection bias. Reassess the estimand and missingness assumptions.", "columns": columns}
            elif kind == "standardize":
                if any(not pd.api.types.is_numeric_dtype(current[c]) for c in columns):
                    raise ValueError("Standardization requires numeric columns.")
                moments = {}
                for c in columns:
                    mean, sd = db.execute(f"SELECT AVG({_quote(c)}), STDDEV_SAMP({_quote(c)}) FROM observations").fetchone()
                    if mean is None or sd is None or not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
                        raise ValueError("Standardization requires finite, nonconstant columns.")
                    moments[c] = {"mean": float(mean), "sample_sd": float(sd)}
                params = []
                expressions = []
                for c in current:
                    if c in moments:
                        expressions.append(f"({_quote(c)} - ?) / ? AS {_quote(c)}")
                        params.extend([moments[c]["mean"], moments[c]["sample_sd"]])
                    else:
                        expressions.append(_quote(c))
                query = "SELECT " + ", ".join(expressions) + " FROM observations"
                current = db.execute(query, params).df()
                details = {"moments": moments, "parameters": params, "warning": "Standardizing a treatment or outcome changes its scale. Binary treatment coding must remain 0/1 for the ATE engine."}
            elif kind == "rename":
                mapping = operation.get("mapping", {})
                if not isinstance(mapping, dict) or not mapping or any(c not in current for c in mapping) or not all(isinstance(c, str) and c for c in mapping.values()):
                    raise ValueError("Supply an existing-column to nonempty-name mapping.")
                names = [mapping.get(c, c) for c in current]
                if len(names) != len(set(names)):
                    raise ValueError("Renaming cannot create duplicate columns.")
                query = "SELECT " + ", ".join(_quote(c) + " AS " + _quote(mapping.get(c, c)) for c in current) + " FROM observations"
                current = db.execute(query).df()
                details = {"mapping": mapping, "warning": "Update the graph, question and temporal roles to match renamed variables."}
            else:
                raise ValueError("Supported operations are drop_missing, standardize and rename.")
            steps.append({"op": kind, "sql": query, "rows_before": before, "rows_after": len(current), "rows_removed": before - len(current), **details})
        finally:
            db.close()
    return current, {"input_sha256": original, "output_sha256": profile(current)["provenance"]["sha256"], "transformations": steps, "rows_before": len(data), "rows_after": len(current), "rows_removed": len(data) - len(current)}
