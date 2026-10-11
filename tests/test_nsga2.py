"""Focused NSGA-II checks; no conversion or equivalence work."""

import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.nsga2 import engine as e
from common import abc
from common.actions import TRANSFORMS
from common.progress import ProgressLogger


class NSGA2Tests(unittest.TestCase):
    def member(self, values, index=0):
        return e.Individual((index,), values)

    def fake_evaluator(self, circuit, recipe, *, budget, **kwargs):
        budget.check()
        budget.tool_calls += 1
        return {"metrics": {"lut_count": sum(recipe), "lut_levels": 100 - sum(recipe)}}

    def config(self, **overrides):
        values = dict(recipe_length=3, population_size=4, offspring_count=3,
                      generations=3, max_tool_calls=100, seed=42, optimization_mode="pareto")
        values.update(overrides)
        return e.Config(**values)

    def test_dominance_and_fronts(self):
        self.assertTrue(e.dominates((1, 2), (1, 3)))
        self.assertFalse(e.dominates((1, 2), (1, 2)))
        self.assertFalse(e.dominates((1, 3), (2, 2)))
        pop = [self.member(v, i) for i, v in enumerate([(1, 3), (2, 2), (3, 3), (4, 4), (1, 3)])]
        fronts = e.nondominated_sort(pop)
        self.assertEqual([[p.recipe[0] for p in f] for f in fronts], [[0, 1, 4], [2], [3]])
        self.assertEqual(e.nondominated_sort([]), [])

    def test_crowding_and_elitist_survival(self):
        pop = [self.member(v, i) for i, v in enumerate([(1, 4), (2, 3), (3, 2), (4, 1)])]
        e.assign_crowding(pop)
        self.assertEqual(pop[0].crowding, float("inf"))
        self.assertAlmostEqual(pop[1].crowding, 4 / 3)
        chosen = e.select_survivors(pop + [self.member((9, 9), 9)], 2)
        self.assertEqual([p.recipe for p in chosen], [(0,), (3,)])
        equal = [self.member((1, 1), i) for i in range(4)]
        e.assign_crowding(equal)
        self.assertEqual([p.crowding for p in equal], [float("inf"), 0, 0, float("inf")])

    def test_variation_and_tournament(self):
        a, b = (0, 0, 0), (15, 15, 15)
        self.assertEqual(e.vary(a, b, random.Random(1), self.config(crossover_rate=0, mutation_rate=0)), [a, b])
        children = e.vary(a, b, random.Random(2), self.config(crossover_rate=1, mutation_rate=0))
        for i in range(3):
            self.assertEqual(sorted(c[i] for c in children), [0, 15])
        children = e.vary(a, b, random.Random(3), self.config(mutation_rate=1))
        self.assertTrue(all(len(c) == 3 and all(0 <= g < len(TRANSFORMS) for g in c) for c in children))
        self.assertNotEqual(children, [a, b])
        first, second = self.member((1, 1)), self.member((2, 2))
        second.rank = 1
        self.assertIs(e.tournament([first, second], random.Random(1)), first)
        self.assertIs(e.TRANSFORMS, TRANSFORMS)

    def test_repeatability_population_and_odd_offspring(self):
        with patch.object(abc, "evaluate_recipe", side_effect=self.fake_evaluator):
            first = e.run("unused.aig", self.config())
            second = e.run("unused.aig", self.config())
        for key in ("best", "pareto", "tool_calls", "cache_hits", "completed_generations"):
            self.assertEqual(first[key], second[key])
        self.assertEqual(first["completed_generations"], 3)
        # Frozen before scalar mode was added: explicit Pareto search is unchanged.
        self.assertEqual(first["best"], {"recipe": [1, 0, 2], "lut_count": 3, "lut_levels": 97})
        self.assertEqual([p["recipe"] for p in first["pareto"]],
                         [[1, 0, 2], [3, 2, 4], [3, 0, 8], [7, 0, 5],
                          [3, 2, 13], [7, 7, 4], [7, 7, 5]])
        self.assertEqual((first["tool_calls"], first["cache_hits"]), (7, 6))
        self.assertEqual(first["successful_evaluations"] + first["cache_hits"], 4 + 3 * 3)
        pairs = [(p["lut_count"], p["lut_levels"]) for p in first["pareto"]]
        self.assertFalse(any(e.dominates(a, b) for a in pairs for b in pairs))
        self.assertEqual(first["best"], min(first["pareto"], key=lambda p: (p["lut_count"], p["lut_levels"], p["recipe"])))

    def test_shared_budget_whole_recipe_and_exhaustion(self):
        seen = []
        def evaluate(*args, **kwargs):
            seen.append(kwargs["budget"])
            return self.fake_evaluator(*args, **kwargs)
        with patch.object(abc, "evaluate_recipe", side_effect=evaluate) as tool:
            result = e.run("unused.aig", self.config(max_tool_calls=2))
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(tool.call_count, 2)
        self.assertTrue(all(b is seen[0] for b in seen))
        self.assertIsNotNone(result["best"])
        with patch.object(abc, "evaluate_recipe", side_effect=self.fake_evaluator) as tool:
            result = e.run("unused.aig", self.config(max_tool_calls=1))
        self.assertEqual(tool.call_count, 1)
        self.assertIsNotNone(result["best"])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["status"], "budget_exhausted")

    def test_repeated_recipe_real_cache_hit(self):
        events = []
        with patch.object(e.random.Random, "randrange", return_value=0), \
             patch.object(abc, "evaluate_recipe", side_effect=self.fake_evaluator) as tool:
            result = e.run("unused.aig", self.config(crossover_rate=0, mutation_rate=0, max_tool_calls=1),
                           progress=events.append)
        self.assertEqual(tool.call_count, 1)
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["best"]["recipe"], [0, 0, 0])
        self.assertEqual(result["cache_hits"], 12)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(events[-1]["cache_hits"], 12)
        self.assertEqual(events[-1]["tool_calls"], 1)

    def test_failures_abort_and_preserve_diagnostics(self):
        for uncertain in (False, True):
            count = 0
            events = []
            def evaluate(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 1:
                    return self.fake_evaluator(*args, **kwargs)
                if uncertain:
                    kwargs["budget"].tool_calls = None
                else:
                    kwargs["budget"].tool_calls += 1
                raise abc.ABCError("timeout or malformed metrics", "raw failure", 1)
            with patch.object(abc, "evaluate_recipe", side_effect=evaluate):
                result = e.run("unused.aig", self.config(), progress=events.append)
            self.assertEqual(count, 2)
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["best"])
            self.assertEqual(result["failure"]["abc_output"], "raw failure")
            self.assertEqual(result["tool_calls"], None if uncertain else 2)
            self.assertEqual(events[-1]["status"], "failed")
            self.assertEqual(events[-1]["tool_calls"], result["tool_calls"])
            with tempfile.TemporaryDirectory() as directory:
                output = e.write_results(result, directory)
                self.assertFalse((output / "best.csv").exists())
                self.assertTrue(json.loads((output / "pareto.json").read_text()))

    def test_output_schema(self):
        with patch.object(abc, "evaluate_recipe", side_effect=self.fake_evaluator):
            result = e.run("unused.aig", self.config())
        with tempfile.TemporaryDirectory() as directory:
            output = e.write_results(result, directory)
            self.assertEqual(json.loads((output / "result.json").read_text()), result)
            self.assertEqual(json.loads((output / "pareto.json").read_text()), result["pareto"])
            with (output / "best.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(json.loads(row["best_recipe"]), result["best"]["recipe"])
            self.assertEqual(json.loads(row["config"]), result["config"])
            self.assertEqual(json.loads(row["provenance"]), result["provenance"])
            self.assertEqual(int(row["tool_calls"]), result["tool_calls"])
            self.assertGreaterEqual(float(row["runtime_seconds"]), 0)

    def test_configuration(self):
        self.assertEqual(asdict(e.load_config()), asdict(e.Config()))
        config = e.load_config()
        self.assertEqual((config.seed, config.population_size, config.offspring_count,
                          config.generations, config.recipe_length, config.max_tool_calls,
                          config.lut_k), (0, 40, 40, 99, 20, 4000, 6))
        self.assertEqual(config.population_size + config.offspring_count * config.generations, 4000)
        self.assertEqual((config.optimization_mode, config.weight_area, config.weight_delay),
                         ("weighted_sum", 0.5, 0.5))
        # Old config files inherit the new default; explicit Pareto remains available.
        values = asdict(config)
        for key in ("optimization_mode", "weight_area", "weight_delay"):
            del values[key]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old-config.yaml"
            path.write_text(json.dumps(values))
            self.assertEqual(e.load_config(path), config)
            values["optimization_mode"] = "pareto"
            path.write_text(json.dumps(values))
            self.assertEqual(e.load_config(path).optimization_mode, "pareto")
        for values in ({"population_size": 1}, {"recipe_length": 0}, {"generations": -1},
                       {"offspring_count": 0}, {"timeout": float("nan")},
                       {"mutation_rate": 2}, {"objective": "weighted_sum"}, {"max_tool_calls": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.config(**values)

    def test_progress_streaming_and_final_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            events = []
            with ProgressLogger(path) as logger, patch("builtins.print") as terminal:
                def progress(event):
                    logger(event)
                    # Read before closing the logger: every JSONL event was flushed.
                    self.assertEqual(json.loads(path.read_text().splitlines()[-1]), event)
                    events.append(event)
                with patch.object(abc, "evaluate_recipe", side_effect=self.fake_evaluator):
                    result = e.run("unused.aig", self.config(), progress=progress)
                self.assertTrue(all(call.kwargs.get("flush") for call in terminal.call_args_list))
            self.assertEqual(events[0]["event"], "configuration")
            self.assertEqual(events[0]["config"], result["config"])
            self.assertEqual([p["generation"] for p in events if p["event"] == "generation"], [0, 1, 2, 3])
            self.assertEqual(events[1]["event"], "improvement")
            self.assertEqual(events[1]["tool_calls"], 1)
            for event in events:
                self.assertEqual(event["budget_utilization_percent"], event["tool_calls"])
                self.assertGreaterEqual(event["elapsed_seconds"], 0)
                self.assertEqual(event["generations"], 3)
            final = events[-1]
            self.assertEqual(final["event"], "summary")
            for key in ("tool_calls", "cache_hits", "status", "best"):
                self.assertEqual(final[key], result[key])
            self.assertEqual(final["pareto_size"], len(result["pareto"]))
            self.assertEqual(result["tool_calls"], result["successful_evaluations"])

    def test_real_cli_overrides_determinism_and_progress(self):
        runner = Path(e.__file__).resolve().parents[2] / "run.py"
        circuit = runner.parent / "benchmarks/converted/iscas85/c17.aig"
        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                process = subprocess.run(
                    [sys.executable, str(runner), "--algorithm", "nsga2", "--circuit", str(circuit),
                     "--optimization-mode", "pareto",
                     "--seed", "7", "--recipe-length", "20", "--population-size", "2",
                     "--offspring-count", "2", "--generations", "1", "--max-tool-calls", "4",
                     "--lut-k", "6", "--crossover-rate", "0.8", "--mutation-rate", "1",
                     "--timeout", "none"], cwd=directory, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                result_path, = Path(directory).glob("results/nsga2/**/result.json")
                result = json.loads(result_path.read_text())
                results.append(result)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["tool_calls"], 4)
                self.assertEqual(result["config"], asdict(e.Config(
                    seed=7, population_size=2, offspring_count=2, generations=1,
                    max_tool_calls=4, crossover_rate=0.8, mutation_rate=1, timeout=None,
                    optimization_mode="pareto")))
                events = [json.loads(line) for line in (result_path.parent / "progress.jsonl").read_text().splitlines()]
                self.assertEqual(events[-1]["tool_calls"], result["tool_calls"])
                self.assertEqual(events[-1]["budget_utilization_percent"], 100)
                self.assertEqual(events[-1]["best"], result["best"])
                self.assertIn("ABC calls=4/4 (100.00%)", process.stdout)
                self.assertIn("generation 1/1", process.stdout)
        for key in ("best", "pareto", "tool_calls", "cache_hits"):
            self.assertEqual(results[0][key], results[1][key])

    def test_cli_failure_and_no_affordable_candidate(self):
        runner = Path(e.__file__).resolve().parents[2] / "run.py"
        circuit = runner.parent / "benchmarks/converted/iscas85/c17.aig"
        for extra, status in ((["--abc-binary", "/nonexistent/abc"], "failed"),
                              (["--max-tool-calls", "0"], "budget_exhausted")):
            with tempfile.TemporaryDirectory() as directory:
                process = subprocess.run(
                    [sys.executable, str(runner), "--algorithm", "nsga2", "--circuit", str(circuit),
                     "--recipe-length", "2", "--population-size", "2", "--generations", "0", *extra],
                    cwd=directory, capture_output=True, text=True, check=False)
                self.assertEqual(process.returncode, 1, process.stderr)
                result_path, = Path(directory).glob("results/nsga2/**/result.json")
                result = json.loads(result_path.read_text())
                self.assertEqual(result["status"], status)
                self.assertEqual(result["tool_calls"], 0)
                events = [json.loads(line) for line in (result_path.parent / "progress.jsonl").read_text().splitlines()]
                self.assertEqual(events[-1]["status"], status)
                self.assertEqual(events[-1]["tool_calls"], result["tool_calls"])
                self.assertIsNone(result["best"])
                self.assertFalse((result_path.parent / "best.csv").exists())

    def test_other_engines_remain_placeholders(self):
        result = subprocess.run([sys.executable, "run.py", "--algorithm", "drills", "--circuit", "missing.aig"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not implemented", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
