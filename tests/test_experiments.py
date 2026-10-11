"""Deterministic configuration identities and safe serial rerun lifecycle."""

from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithms.nsga2 import engine as e
from common import experiments as x
import run as runner


class ExperimentTests(unittest.TestCase):
    circuit = ROOT / "benchmarks/converted/iscas85/c17.aig"

    def describe(self, config=None, circuit=None):
        return x.describe_experiment("nsga2", circuit or self.circuit, asdict(config or e.Config()))

    def test_stable_path_and_seed_siblings(self):
        experiment = self.describe()
        first = x.experiment_path(experiment)
        self.assertEqual(first, x.experiment_path(self.describe()))
        reversed_config = dict(reversed(list(experiment["config"].items())))
        self.assertEqual(first, x.experiment_path({**experiment, "config": reversed_config}))
        sibling = x.experiment_path(self.describe(replace(e.Config(), seed=1)))
        self.assertEqual(first.parent, sibling.parent)
        self.assertEqual((first.name, sibling.name), ("seed_0", "seed_1"))
        self.assertEqual(first.parts[:4], ("results", "nsga2", "iscas85", "c17"))
        self.assertEqual(first.parent.name, "pop40_off40_gen99_len20")
        epfl = x.experiment_path(self.describe(circuit=ROOT / "benchmarks/epfl/arithmetic/adder.aig"))
        self.assertEqual(epfl.parts[:4], ("results", "nsga2", "epfl", "adder"))

    def test_only_declared_parameters_and_seed_affect_config_path(self):
        original = self.describe()
        first = x.experiment_path(original)
        changes = dict(population_size=41, offspring_count=41, generations=98, recipe_length=21,
                       checkpoint_interval=100, max_tool_calls=3999, crossover_rate=0.8, mutation_rate=0.04, lut_k=5,
                       timeout=None, weight_area=0.6, weight_delay=0.4, objective="other",
                       abc_binary="another-abc", optimization_mode="pareto", future_parameter="value")
        for key, value in changes.items():
            with self.subTest(key=key):
                experiment = {**original, "config": {**original["config"], key: value}}
                if key in dict(x.PARAMETERS):
                    self.assertNotEqual(first, x.experiment_path(experiment))
                else:
                    self.assertEqual(first, x.experiment_path(experiment))
        changed_binary = {**original, "abc_executable": {**original["abc_executable"], "resolved": "/other/abc"}}
        self.assertEqual(first, x.experiment_path(changed_binary))
        near_float = {**original, "config": {**original["config"], "crossover_rate": 0.9000000000000001}}
        self.assertEqual(first, x.experiment_path(near_float))

    def test_benchmark_and_circuit_stem_paths(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(x, "REPOSITORY", Path(directory)):
            first = self.describe(circuit=Path(directory) / "suite_a/c17.aig")
            second = self.describe(circuit=Path(directory) / "suite_b/c17.aig")
            extension = self.describe(circuit=Path(directory) / "suite_a/c17.blif")
            self.assertNotEqual(x.experiment_path(first), x.experiment_path(second))
            self.assertEqual(x.experiment_path(first), x.experiment_path(extension))
            self.assertEqual(first["circuit_identity"], "repo/suite_a/c17.aig")
            external = self.describe(circuit="/suite_a/c17.aig")
            self.assertTrue(external["circuit_identity"].startswith("external/"))

    def test_fixed_settings_do_not_lengthen_paths(self):
        paths = []
        for binary in ("a_b", "a%5Fb", "a/b", "a b", "..", "a"*500):
            experiment = self.describe(replace(e.Config(), abc_binary=binary))
            path = x.experiment_path(experiment)
            paths.append(path)
            self.assertTrue(all(len(part.encode()) <= 200 for part in path.parts))
            self.assertNotIn("..", path.parts)
            self.assertEqual(path, x.experiment_path(experiment))
        self.assertEqual(len(set(paths)), 1)

    def test_prepare_clears_only_run_owned_files(self):
        with tempfile.TemporaryDirectory() as root:
            experiment = self.describe()
            directory = x.prepare_run(experiment, root)
            for name in x.RUN_ARTIFACTS:
                (directory / name).write_text("old success")
            (directory / "notes.txt").write_text("user notes")
            other_seed = x.prepare_run(self.describe(replace(e.Config(), seed=1)), root)
            (other_seed / "result.json").write_text("other seed")
            other_config = x.prepare_run(self.describe(replace(e.Config(), population_size=41)), root)
            (other_config / "result.json").write_text("other configuration")
            self.assertEqual(x.prepare_run(experiment, root), directory)
            for name in set(x.RUN_ARTIFACTS) - {"config.json"}:
                self.assertFalse((directory / name).exists())
            self.assertEqual(json.loads((directory / "config.json").read_text()), experiment)
            self.assertEqual((directory / "notes.txt").read_text(), "user notes")
            self.assertEqual((other_seed / "result.json").read_text(), "other seed")
            self.assertEqual((other_config / "result.json").read_text(), "other configuration")

    def test_cleanup_refuses_directory_symlinks_and_artifact_directories(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as unrelated:
            experiment = self.describe()
            directory = x.experiment_path(experiment, root)
            directory.parent.mkdir(parents=True)
            directory.symlink_to(unrelated, target_is_directory=True)
            preserved = Path(unrelated) / "result.json"
            preserved.write_text("unrelated")
            with self.assertRaisesRegex(ValueError, "symlink"):
                x.prepare_run(experiment, root)
            self.assertEqual(preserved.read_text(), "unrelated")
            directory.unlink()
            directory.mkdir()
            (directory / "best.csv").mkdir()
            (directory / "result.json").write_text("preserved on failed preflight")
            with self.assertRaisesRegex(ValueError, "not a file"):
                x.prepare_run(experiment, root)
            self.assertTrue((directory / "result.json").exists())

    def test_cli_reruns_clear_success_before_execution_and_keep_effective_config(self):
        config = e.Config(population_size=2, offspring_count=2, generations=0, max_tool_calls=3)
        experiment = self.describe(config)
        argv = ["run.py", "--algorithm", "nsga2", "--circuit", str(self.circuit),
                "--population-size", "2", "--offspring-count", "2", "--generations", "0", "--max-tool-calls", "3"]
        prepare = x.prepare_run
        with tempfile.TemporaryDirectory() as root, patch.object(sys, "argv", argv), \
             patch.object(x, "prepare_run", side_effect=lambda description: prepare(description, root)), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 0)
            directory = x.experiment_path(experiment, root)
            original_result = json.loads((directory / "result.json").read_text())
            self.assertTrue((directory / "best.csv").exists())
            (directory / "notes.txt").write_text("keep")
            # At run entry an interruption must find no old successful result.
            def interrupt(*args, **kwargs):
                for name in ("result.json", "best.csv", "best.json", "pareto.json"):
                    self.assertFalse((directory / name).exists())
                self.assertTrue((directory / "config.json").exists())
                raise KeyboardInterrupt
            with patch.object(e, "run", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                runner.main()
            self.assertFalse((directory / "result.json").exists())
            # Restore a success, then fail the same experiment after ABC launch.
            self.assertEqual(runner.main(), 0)
            rerun = json.loads((directory / "result.json").read_text())
            self.assertEqual(original_result["best"], rerun["best"])
            process = Mock()
            process.communicate.side_effect = [subprocess.TimeoutExpired("abc", 60), ("", None)]
            with patch.object(e.abc.subprocess, "Popen", return_value=process), \
                 patch.object(sys, "stderr", io.StringIO()):
                self.assertEqual(runner.main(), 1)
            result = json.loads((directory / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["tool_calls"], 1)
            self.assertFalse((directory / "best.csv").exists())
            self.assertFalse((directory / "pareto.json").exists())
            self.assertIsNone(json.loads((directory / "best.json").read_text()))
            self.assertEqual((directory / "notes.txt").read_text(), "keep")
            metadata = json.loads((directory / "config.json").read_text())
            self.assertEqual(metadata, experiment)
            self.assertEqual(metadata["config"], result["config"])
            self.assertEqual(metadata["abc_executable"]["resolved"], result["provenance"]["abc_executable"])
            self.assertEqual(metadata["abc_executable"]["sha256"], result["provenance"]["abc_sha256"])
            events = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["status"], "failed")
            self.assertEqual(events[-1]["tool_calls"], 1)
            self.assertTrue(all(event["results_directory"] == str(directory) for event in events))
            self.assertTrue(all(event["effective_config"] == asdict(config) for event in events))

    def test_library_write_results_reuses_path_and_removes_stale_success(self):
        def evaluate(circuit, recipe, *, budget, **kwargs):
            budget.check()
            budget.tool_calls += 1
            return {"metrics": {"lut_count": 2, "lut_levels": 1}}
        with patch.object(e.abc, "evaluate_recipe", side_effect=evaluate):
            result = e.run(self.circuit, e.Config(optimization_mode="pareto", population_size=2,
                                                generations=0, max_tool_calls=2))
        with tempfile.TemporaryDirectory() as root:
            first = e.write_results(result, root)
            (first / "notes.txt").write_text("keep")
            failed = {**result, "status": "failed", "best": None, "failure": {"message": "failed"}}
            second = e.write_results(failed, root)
            self.assertEqual(first, second)
            self.assertFalse((second / "best.csv").exists())
            self.assertEqual((second / "notes.txt").read_text(), "keep")
            self.assertEqual(json.loads((second / "result.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
