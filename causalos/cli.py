"""Command line for reproducible local analysis and the interactive laboratory."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

from causalos import __version__
from causalos.report import build_report, clean, render_html
from causalos.runs import create_run, read_run, replay_run, publish_new, read_csv_table, table_frame, table_snapshot


def _main():
    parser = argparse.ArgumentParser(prog="causalos", description="CAUSALOS · causal evidence laboratory")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Launch the local web laboratory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    for name in ("analyze", "profile"):
        item = sub.add_parser(name)
        item.add_argument("csv", type=Path)
        item.add_argument("--out", type=Path)
        if name == "analyze":
            item.add_argument("--question", type=Path, required=True)
            item.add_argument("--report", type=Path)
            item.add_argument("--include-data", action="store_true", help="Explicitly embed raw scalar rows in the saved run")
    for name in ("run-open", "run-replay"):
        item = sub.add_parser(name, help="Inspect or numerically replay a saved analysis run")
        item.add_argument("run", type=Path)
        item.add_argument("--csv", type=Path, help="Matching original scalar dataset for replay")
        item.add_argument("--out", type=Path)
        item.add_argument("--report", type=Path)
    example = sub.add_parser("example", help="Export a reproducible dataset and question")
    example.add_argument("name", nargs="?", default="confounded")
    example.add_argument("--seed", type=int, default=42)
    example.add_argument("--out", type=Path, default=Path("artifacts/example"))
    replay = sub.add_parser("replay", help="Run the simulated sequential experiment")
    replay.add_argument("--seed", type=int, default=42)
    replay.add_argument("--steps", type=int, default=5)
    replay.add_argument("--out", type=Path)
    bench = sub.add_parser("benchmark", help="Execute train/validation/test policy evaluation")
    bench.add_argument("--out", type=Path, default=Path("artifacts/benchmark"))
    bench.add_argument("--study", choices=['nominal', 'evidence'], default='nominal')
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("causalos.api:app", host=args.host, port=args.port)
        return
    if args.command in ("run-open", "run-replay"):
        run = read_run(args.run)
        verification = "unverified_saved_result"
        result = {"run": run.bundle(), "verification": verification,
                  "contains_raw_data": run.bundle(include_data=True)["data"] is not None}
        if args.command == "run-replay":
            data = read_csv_table(args.csv) if args.csv else None
            result = replay_run(run, data)
            if result["status"] == "matched":
                verification = "replayed_in_this_process"
        if args.report:
            publish_new(args.report, render_html(build_report(run=run, run_verification=verification)))
        text = json.dumps(clean(result), indent=2, allow_nan=False)
        if args.out:
            publish_new(args.out, text)
        else:
            print(text)
        return 0 if args.command == "run-open" or result["status"] == "matched" else 2
    from causalos import datasets, design, profiling
    if args.command == "example":
        data, metadata = datasets.example(args.name, seed=args.seed)
        args.out.mkdir(parents=True, exist_ok=False)
        data.to_csv(args.out / "data.csv", index=False)
        (args.out / "question.json").write_text(json.dumps(clean(metadata.get("question", {})), indent=2))
        (args.out / "provenance.json").write_text(json.dumps(clean(metadata), indent=2))
        print(args.out.resolve())
        return
    if args.command == "benchmark":
        if args.study == 'evidence':
            from causalos.evidence_design import run_benchmark
        else:
            from causalos.benchmark import run_benchmark
        run_benchmark(args.out)
        print(args.out.resolve())
        return
    elif args.command == "replay":
        result = design.initial_state(seed=args.seed)
        for _ in range(max(0, min(args.steps, 100))):
            rankings = design.recommend(result).get("ranked", [])
            if not rankings:
                break
            result = design.step(result, rankings[0]["id"], seed=args.seed)
    else:
        data = pd.read_csv(args.csv, float_precision="round_trip") if args.command == "profile" else read_csv_table(args.csv)
        if args.command == "profile":
            result = profiling.profile(data)
        else:
            question = json.loads(args.question.read_text())
            run = create_run(data, question, include_data=args.include_data)
            result = run.bundle(include_data=args.include_data)
            if args.report:
                metadata_path = args.csv.parent / 'provenance.json'
                metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {'source_file': args.csv.name}
                args.report.parent.mkdir(parents=True, exist_ok=True)
                publish_new(args.report, render_html(build_report(run=run, run_verification="computed_in_this_process", profile=profiling.profile(table_frame(table_snapshot(data))), metadata=metadata)))
    text = json.dumps(clean(result), indent=2, allow_nan=False)
    if getattr(args, "out", None):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        publish_new(args.out, text)
        print(args.out.resolve())
    else:
        print(text)


def main():
    try:
        return _main()
    except (ValueError, OSError, TypeError, KeyError, csv.Error) as exc:
        print(f"causalos: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
