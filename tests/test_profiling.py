import duckdb
import pandas as pd
import pytest
from pathlib import Path
from causalos.profiling import profile
from causalos.datasets import example


def test_missingness_duplicates_schema_and_provenance():
    data = pd.DataFrame({"T": [0, 1, 1], "X": [1.0, None, None], 'odd";name': [1, 2, 2]})
    p = profile(data, "T", ["X"])
    assert p["rows"] == 3 and p["duplicates"] == 1
    assert p["columns"][1]["missing"] == 2
    assert p["provenance"]["rows_removed"] == 0
    assert p["provenance"]["transformations"] == []
    assert p["provenance"]["sha256"] == profile(data)["provenance"]["sha256"]
    assert p["provenance"]["sha256"] != profile(data.iloc[::-1])["provenance"]["sha256"]
    assert any("COUNT" in s for s in p["provenance"]["queries"])


def test_sql_cohort_queries_on_actual_data():
    observations, _ = example("nsw")
    root = Path(__file__).resolve().parents[1]
    db = duckdb.connect(":memory:")
    db.register("observations", observations)
    try:
        for name in ["profile.sql", "cohorts.sql"]:
            result = db.execute((root / "sql" / name).read_text()).fetchall()
            assert result
        summary = db.execute("SELECT treat,COUNT(*) FROM observations GROUP BY treat ORDER BY treat").fetchall()
        assert summary == [(0, 260), (1, 185)]
        assert sum(c["n"] for c in profile(observations, "treat")["cohorts"]) == 445
    finally:
        db.close()


def test_reject_duplicate_columns():
    with pytest.raises(ValueError, match="unique"):
        profile(pd.DataFrame([[1, 2]], columns=["X", "X"]))


def test_explicit_sql_transformations_preserve_audit():
    from causalos.profiling import audited_transform
    source = pd.DataFrame({"pre": [1.0, 2.0, 3.0, None], "T": [0, 1, 0, 1]})
    original = source.copy()
    changed, audit = audited_transform(source, [
        {"op": "drop_missing", "columns": ["pre"]},
        {"op": "standardize", "columns": ["pre"]},
        {"op": "rename", "mapping": {"pre": "baseline"}},
    ])
    assert source.equals(original)
    assert len(changed) == 3 and abs(changed.baseline.mean()) < 1e-12
    assert abs(changed.baseline.std() - 1) < 1e-12
    assert audit["rows_removed"] == 1 and len(audit["transformations"]) == 3
    assert audit["input_sha256"] != audit["output_sha256"]
    assert all("sql" in step for step in audit["transformations"])
    with pytest.raises(ValueError, match="Supported"):
        audited_transform(source, [{"op": "execute", "sql": "DROP TABLE observations"}])
