"""AST-GWO numerical fidelity, shared evaluation, and CLI contracts."""
import ast
from dataclasses import replace
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithms.ast_gwo import engine as e
from algorithms.ast_gwo.optimizer import Optimizer, Leaders, bits_to_recipe
from common.experiments import describe_experiment, experiment_path

UPSTREAM = ROOT.parent / 'ast-gwo-aicas2026/algorithms/ast_gwo/ast_gwo.py'


class CoreTests(unittest.TestCase):
    def test_initialization_and_opcode_order(self):
        cfg = e.Config(population_size=5, recipe_length=20, seed=7)
        first, second = Optimizer(cfg), Optimizer(cfg)
        expected = np.random.default_rng(7).integers(0, 2, (5, 80), dtype=np.uint8)
        np.testing.assert_array_equal(first.pop, expected)
        np.testing.assert_array_equal(first.pop, second.pop)
        bits = np.array([int(b) for i in range(16) for b in f'{i:04b}'], dtype=np.uint8)
        self.assertEqual(bits_to_recipe(bits), tuple(range(16)))
        self.assertEqual(bits_to_recipe(np.tile([1, 0, 1, 0], 3)), (10, 10, 10))
        for row in first.pop:
            recipe = bits_to_recipe(row)
            self.assertEqual(len(recipe), 20)
            self.assertTrue(all(type(i) is int and 0 <= i < 16 for i in recipe))
        for invalid in ([], [0], [0, 0, 0, 2], [[0, 0, 0, 1]]):
            with self.assertRaises(ValueError):
                bits_to_recipe(invalid)

    def test_coefficient_schedule(self):
        opt = Optimizer(e.Config(generations=10))
        for t, a, k in ((0, 2, 6), (5, 1, 10), (10, 0, 14)):
            opt._update_coefficients(t)
            self.assertEqual((opt.a, opt.k), (a, k))

    def test_controlled_position_and_sigmoid_threshold(self):
        opt = Optimizer(e.Config(population_size=3, recipe_length=1))
        opt.pop[:] = [0, 1, 0, 1]
        opt._leaders.bits = [np.array([0, 1, 0, 1], dtype=np.uint8)] * 3
        # A=0, so X_mean is the leader. S(0)~0.047, S(1)~0.953.
        draws = [np.full(opt.pop.shape, .5)] * 6 + [np.full(opt.pop.shape, .1)]
        opt._rng = SimpleNamespace(random=lambda shape: draws.pop(0))
        opt._step()
        np.testing.assert_array_equal(opt.pop, np.tile([0, 1, 0, 1], (3, 1)))
        self.assertEqual(opt.pop.dtype, np.uint8)
        self.assertFalse(draws)

    def test_historical_leaders_and_duplicates(self):
        leaders = Leaders()
        pop = np.eye(4, dtype=np.uint8)
        fitness = np.array([4., 1., 3., 2.])
        leaders.update(pop, fitness, -fitness)
        self.assertEqual(leaders.fit, [1, 2, 3])
        np.testing.assert_array_equal(leaders.alpha, pop[1])
        pop[1] = 0  # leaders own their data
        self.assertEqual(int(leaders.alpha.sum()), 1)
        leaders.update(pop, fitness, -fitness)
        self.assertEqual(leaders.fit, [1, 1, 2])
        self.assertEqual(leaders.alpha_improve, -1)

    @unittest.skipUnless(UPSTREAM.is_file(), 'optional local upstream source absent')
    def test_actual_upstream_numerical_operations(self):
        # Compile the original classes without imports/ABC initialization. This
        # executes the actual local source and never writes to that repository.
        tree = ast.parse(UPSTREAM.read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name in ('Leaders', 'AstGWO')]
        namespace = {'np': np}
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), *classes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(UPSTREAM), 'exec'), namespace)
        for seed in (0, 7, 42):
            cfg = e.Config(population_size=5, recipe_length=3, generations=8, seed=seed)
            port = Optimizer(cfg)
            original = namespace['AstGWO'].__new__(namespace['AstGWO'])
            original.cfg = SimpleNamespace(iters=8, k_min=6., k_max=14.)
            original._rng = np.random.default_rng(seed)
            original.pop = original._rng.integers(0, 2, (5, 12), dtype=np.uint8)
            original._leaders = namespace['Leaders']()
            original.a, original.k = 2., 6.
            for t in range(9):
                if t:
                    # Compare the floating-point sigmoid inputs too: binary
                    # equality alone could conceal a numerical perturbation.
                    exp = np.exp
                    sigmoid_inputs = []
                    def record_exp(value):
                        sigmoid_inputs.append(value.copy())
                        return exp(value)
                    with patch.object(np, 'exp', side_effect=record_exp):
                        original._step()
                        port._step()
                    np.testing.assert_array_equal(sigmoid_inputs[0], sigmoid_inputs[1])
                    original._update_coefficients(t)
                    port._update_coefficients(t)
                np.testing.assert_array_equal(port.pop, original.pop)
                self.assertEqual((port.a, port.k), (original.a, original.k))
                fit = np.random.default_rng(t).integers(0, 4, 5).astype(float)
                for optimizer in (original, port):
                    optimizer._leaders.update(optimizer.pop, fit, 100*(1-fit))
                np.testing.assert_array_equal(port._leaders.bits, original._leaders.bits)
                self.assertEqual(port._leaders.fit, original._leaders.fit)


class EngineTests(unittest.TestCase):
    circuit = ROOT / 'benchmarks/converted/iscas85/c17.aig'

    def execute(self, config=None, population=None, error_at=None, metrics=None):
        events, recipes = [], []
        def evaluate(circuit, recipe, *, budget, **kwargs):
            self.assertEqual(circuit, self.circuit)
            self.assertEqual(kwargs['metric'], 'lut')
            self.assertEqual(kwargs['lut_k'], 6)
            budget.check()
            budget.tool_calls += 1
            budget.abc_process_invocations += 1
            recipes.append(tuple(recipe))
            if len(recipes) == error_at:
                raise e.abc.ABCError('timed out', output='diagnostic')
            value = (100, 10) if not recipe else (metrics(recipe) if metrics else (100-sum(recipe), 5))
            return {'metrics': dict(zip(('lut_count', 'lut_levels'), value))}
        config = config or e.Config(population_size=3, recipe_length=1, generations=0)
        optimizer = Optimizer(config)
        if population is not None:
            optimizer.pop = np.array([[int(b) for i in recipe for b in f'{i:04b}']
                                      for recipe in population], dtype=np.uint8)
        with patch.object(e.abc, 'evaluate_recipe', side_effect=evaluate), \
             patch.object(e, 'Optimizer', return_value=optimizer):
            result = e.run(self.circuit, config, progress=events.append)
        return result, events, recipes

    def test_weighted_fitness_and_best(self):
        pop = [(0,), (1,), (2,)]
        score = lambda r: (20, 10) if r[0] == 1 else (100, 2)
        for area, target in ((.8, 1), (.2, 0)):
            cfg = e.Config(population_size=3, recipe_length=1, generations=0,
                           weight_area=area, weight_delay=1-area)
            result, events, _ = self.execute(cfg, pop, metrics=score)
            self.assertEqual(result['best']['recipe'], [target])
            self.assertAlmostEqual(result['weighted_fitness'], .36)
            self.assertEqual(result['best'], events[-1]['best'])

    def test_cache_at_full_budget(self):
        cfg = e.Config(population_size=3, recipe_length=2, generations=0, max_tool_calls=1)
        result, _, recipes = self.execute(cfg, [(4, 4)]*3)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(recipes, [(), (4, 4)])
        self.assertEqual((result['tool_calls'], result['evaluations'], result['cache_hits']), (1, 1, 2))
        self.assertEqual(result['baseline_tool_calls'], 1)
        self.assertEqual(result['abc_process_invocations'], 2)

    def test_partial_initial_population_and_zero_budget(self):
        for budget in (0, 1, 2):
            cfg = e.Config(population_size=3, recipe_length=1, generations=2, max_tool_calls=budget)
            result, events, recipes = self.execute(cfg, [(0,), (1,), (2,)])
            self.assertEqual(result['status'], 'budget_exhausted')
            self.assertEqual(result['population_evaluated'], budget)
            self.assertEqual(result['tool_calls'], budget)
            self.assertEqual(result['candidate_attempts'], budget+1)
            self.assertEqual(result['completed_iterations'], 0)
            self.assertEqual(result['best']['recipe'] if budget else None, [budget-1] if budget else None)
            self.assertEqual(events[-1]['best'], result['best'])
            self.assertEqual(len(recipes), budget+1)

    def test_partial_later_iteration_and_coefficient_order(self):
        cfg = e.Config(population_size=3, recipe_length=2, generations=4, max_tool_calls=4)
        calls = []
        def step(opt):
            calls.append((opt.a, opt.k))
            opt.pop = np.array([[int(b) for i in (j, j) for b in f'{i:04b}']
                                for j in (3, 4, 5)], dtype=np.uint8)
        with patch.object(Optimizer, '_step', step):
            result, _, _ = self.execute(cfg, [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(calls, [(2, 6)])
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['generation'], 1)
        self.assertEqual(result['completed_iterations'], 0)
        self.assertEqual(result['population_evaluated'], 1)
        self.assertEqual(result['best']['recipe'], [3, 3])
        self.assertEqual((result['a'], result['k']), (1.5, 8))

    def test_duplicates_do_not_end_iterations(self):
        cfg = e.Config(population_size=3, recipe_length=1, generations=4, max_tool_calls=1)
        with patch.object(Optimizer, '_step', lambda opt: None):
            result, events, _ = self.execute(cfg, [(2,)]*3)
        self.assertEqual(result['completed_iterations'], 4)
        self.assertEqual(result['cache_hits'], 14)
        self.assertEqual(len([v for v in events if v['event'] == 'generation']), 5)

    def test_errors_abort_and_retain_best_so_far(self):
        result, _, _ = self.execute(population=[(0,), (1,), (2,)], error_at=3)
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['best'])
        self.assertEqual(result['best_so_far']['recipe'], [0])
        self.assertEqual(result['tool_calls'], 2)
        self.assertEqual(result['failure']['abc_output'], 'diagnostic')
        for metrics in ((float('inf'), 1), (-1, 1), (1, None)):
            result, _, _ = self.execute(metrics=lambda r: metrics)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['evaluations'], 0)
        result, _, _ = self.execute(error_at=1)
        self.assertEqual(result['tool_calls'], 0)
        self.assertEqual(result['baseline_tool_calls'], 1)

    def test_deterministic_run_and_checkpoints(self):
        cfg = e.Config(population_size=5, recipe_length=2, generations=3, checkpoint_interval=2)
        first, events, _ = self.execute(cfg)
        second, _, _ = self.execute(cfg)
        self.assertEqual(first['best'], second['best'])
        for key in ('tool_calls', 'cache_hits', 'evaluations', 'candidate_attempts'):
            self.assertEqual(first[key], second[key])
        checkpoints = [v for v in events if v['event'] == 'checkpoint']
        thresholds = [n for v in checkpoints for n in v['checkpoint_thresholds']]
        self.assertEqual(thresholds, list(range(2, first['tool_calls']+1, 2)))
        self.assertEqual(events[-1]['best_so_far'], first['best'])

    def test_config_validation(self):
        for values in ({'population_size': 2}, {'generations': -1}, {'seed': -1},
                       {'k_min': 0}, {'k_max': 5}, {'k_min': float('nan')},
                       {'k_max': float('inf')}, {'optimization_mode': 'pareto'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                e.Config(**values)
        self.assertEqual(e.load_config().population_size, 50)
        self.assertEqual(e.load_config(k_min=18, k_max=18).k_max, 18)

    @unittest.skipUnless(shutil.which('yosys-abc'), 'ABC unavailable')
    def test_real_cli_serialization_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory)/'override.json'
            config_path.write_text(json.dumps({'population_size': 4, 'recipe_length': 3,
                'generations': 3, 'max_tool_calls': 5, 'seed': 99}))
            command = [sys.executable, str(ROOT/'run.py'), '--algorithm', 'ast_gwo',
                       '--circuit', str(self.circuit), '--config', str(config_path),
                       '--seed', '7', '--k-min', '8', '--k-max', '12', '--timeout', 'none']
            previous = None
            for _ in range(2):
                process = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
                self.assertEqual(process.returncode, 0, process.stderr)
                cfg = e.load_config(config_path, seed=7, k_min=8, k_max=12)
                cfg = replace(cfg, timeout=None)
                path = experiment_path(describe_experiment('ast_gwo', self.circuit, e.asdict(cfg)),
                                       Path(directory)/'results')
                result = json.loads((path/'result.json').read_text())
                self.assertEqual(result['status'], 'budget_exhausted')
                self.assertEqual(result['tool_calls'], 5)
                self.assertEqual(result['baseline_tool_calls'], 1)
                self.assertEqual(result['config'], e.asdict(cfg))
                self.assertEqual(json.loads((path/'best.json').read_text()), result['best'])
                with (path/'best.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(json.loads(rows[0]['best_recipe']), result['best']['recipe'])
                self.assertEqual(json.loads((path/'config.json').read_text()), result['experiment'])
                events = [json.loads(line) for line in (path/'progress.jsonl').read_text().splitlines()]
                self.assertEqual(events[-1]['best'], result['best'])
                self.assertEqual(result['provenance']['bit_dimension'], 12)
                if previous:
                    self.assertEqual(previous['best'], result['best'])
                    self.assertEqual(previous['tool_calls'], result['tool_calls'])
                previous = result
            invalid = subprocess.run(command+['--population-size', '2'], cwd=directory,
                                     capture_output=True, text=True)
            self.assertEqual(invalid.returncode, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
