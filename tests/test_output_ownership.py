import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from causalos import benchmark, evidence_design


class OutputOwnership(unittest.TestCase):
    def test_example_refuses_existing_preserves_and_fresh_run_works(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "existing"
            target.mkdir()
            names = ["data.csv", "question.json", "provenance.json"]
            for name in names:
                (target / name).write_text("original-" + name)
            command = [sys.executable, "-m", "causalos.cli", "example", "confounded", "--out"]
            p = subprocess.run(command + [str(target)], text=True, capture_output=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertEqual(
                {n: (target / n).read_text() for n in names}, {n: "original-" + n for n in names}
            )
            new = root / "fresh"
            p = subprocess.run(command + [str(new)], text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual({x.name for x in new.iterdir()}, set(names))
            self.assertTrue(json.loads((new / "question.json").read_text()))
            snapshot = {n: (new / n).read_bytes() for n in names}
            p = subprocess.run(command + [str(new)], text=True, capture_output=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertEqual(snapshot, {n: (new / n).read_bytes() for n in names})

    def test_both_benchmark_entrypoints_refuse_before_any_work(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for label, module, attr in [
                ("nominal", benchmark, "optimize"),
                ("evidence", evidence_design, "episode"),
            ]:
                target = root / label
                target.mkdir()
                (target / "summary.json").write_text("original")
                with (
                    self.subTest(label=label),
                    patch.object(module, attr, side_effect=AssertionError("expensive work started")),
                ):
                    with self.assertRaises(FileExistsError):
                        module.run_benchmark(target)
                self.assertEqual((target / "summary.json").read_text(), "original")

    def test_symlink_and_regular_file_refused_before_benchmark(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "other"
            folder.mkdir()
            sentinel = folder / "config.json"
            sentinel.write_text("original")
            symlink = root / "link"
            symlink.symlink_to(folder, target_is_directory=True)
            file = root / "file"
            file.write_text("original file")
            for path in [symlink, file]:
                with (
                    self.subTest(path=path),
                    patch.object(benchmark, "optimize", side_effect=AssertionError("started")),
                ):
                    with self.assertRaises(FileExistsError):
                        benchmark.run_benchmark(path)
            self.assertEqual(sentinel.read_text(), "original")
            self.assertEqual(file.read_text(), "original file")

    def test_simultaneous_example_publishers_have_one_owner(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "shared"
            command = [sys.executable, "-m", "causalos.cli", "example", "confounded", "--out", str(target)]
            processes = [
                subprocess.Popen(
                    command + ["--seed", str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                for seed in (17, 23)
            ]
            outputs = [p.communicate(timeout=60) for p in processes]
            self.assertCountEqual([p.returncode for p in processes], [0, 1], outputs)
            names = {"data.csv", "question.json", "provenance.json"}
            self.assertEqual({p.name for p in target.iterdir()}, names)
            self.assertTrue(json.loads((target / "question.json").read_text()))
            self.assertTrue(json.loads((target / "provenance.json").read_text()))
            snapshot = {n: (target / n).read_bytes() for n in names}
            retry = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(retry.returncode, 1)
            self.assertEqual(snapshot, {n: (target / n).read_bytes() for n in names})

    def test_fresh_benchmark_direct_calls_reserve_then_begin_work(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for label, module, attr in [
                ("nominal", benchmark, "optimize"),
                ("evidence", evidence_design, "episode"),
            ]:
                target = root / label
                with (
                    self.subTest(label=label),
                    patch.object(module, attr, side_effect=RuntimeError("owned controlled stop")),
                ):
                    with self.assertRaisesRegex(RuntimeError, "owned controlled stop"):
                        module.run_benchmark(target)
                self.assertTrue(target.is_dir())
                self.assertEqual(list(target.iterdir()), [])

    def test_real_cli_benchmark_and_example_reject_symlink_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            owned = root / "owned"
            owned.mkdir()
            sentinel = owned / "summary.json"
            sentinel.write_text("preserve")
            link = root / "link"
            link.symlink_to(owned, target_is_directory=True)
            commands = [["example", "confounded"], ["benchmark"], ["benchmark", "--study", "evidence"]]
            for args in commands:
                with self.subTest(args=args):
                    p = subprocess.run(
                        [sys.executable, "-m", "causalos.cli", *args, "--out", str(link)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(p.returncode, 1, (p.stdout, p.stderr))
                    self.assertIn("causalos:", p.stderr)
                    self.assertEqual({p.name for p in owned.iterdir()}, {"summary.json"})
                    self.assertEqual(sentinel.read_text(), "preserve")
                    self.assertTrue(link.is_symlink())
