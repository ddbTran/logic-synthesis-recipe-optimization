"""AST-GWO baseline with shared recipe evaluation and run artifacts."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from algorithms.greedy.engine import Config as SharedConfig, RecipeEvaluator, valid_metrics
from algorithms.nsga2.engine import fingerprint, write_results
from common import abc
from common.actions import TRANSFORMS
from common.progress import CheckpointScheduler
from .optimizer import Optimizer, bits_to_recipe


@dataclass(frozen=True)
class Config(SharedConfig):
    population_size: int = 50
    generations: int = 200
    k_min: float = 6.0
    k_max: float = 14.0

    def __post_init__(self):
        if self.optimization_mode != "weighted_sum":
            raise ValueError("AST-GWO supports only optimization_mode weighted_sum")
        super().__post_init__()
        for name, minimum in (("population_size", 3), ("generations", 0), ("seed", 0)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("k_min", "k_max"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.k_min <= 0 or self.k_max < self.k_min:
            raise ValueError("Require 0 < k_min <= k_max")


def load_config(path=None, **overrides):
    path = Path(path) if path else Path(__file__).with_name("config.yaml")
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object (YAML subset)")
    values.update({key: value for key, value in overrides.items() if value is not None})
    return Config(**values)


def run(circuit, config, *, progress=None):
    started = time.perf_counter()
    budget, baseline_budget = abc.ActionBudget(config.max_tool_calls), abc.ActionBudget(1)
    evaluator = RecipeEvaluator(circuit, config, None, budget)
    checkpoints = CheckpointScheduler(config.checkpoint_interval)
    baseline = best = None
    baseline_evaluations = iteration = completed_iterations = population_evaluated = 0
    optimizer = Optimizer(config)
    status, failure = "completed", None

    def snapshot():
        return {
            "algorithm": "ast_gwo", "circuit": str(circuit),
            "generation": iteration, "generations": config.generations,
            "completed_iterations": completed_iterations,
            "population_evaluated": population_evaluated,
            "a": optimizer.a, "k": optimizer.k,
            "optimization_mode": config.optimization_mode,
            "weight_area": config.weight_area, "weight_delay": config.weight_delay,
            "baseline_qor": baseline, "baseline_evaluations": baseline_evaluations,
            "baseline_tool_calls": baseline_budget.tool_calls,
            "abc_process_invocations": budget.abc_process_invocations + baseline_budget.abc_process_invocations,
            "best": best, "best_so_far": best,
            "weighted_fitness": best["weighted_fitness"] if best else None,
            "best_fitness": best["weighted_fitness"] if best else None,
            "best_lut_count": best["lut_count"] if best else None,
            "best_lut_levels": best["lut_levels"] if best else None,
            "evaluations": evaluator.evaluations, "successful_evaluations": evaluator.evaluations,
            "candidate_attempts": evaluator.candidate_attempts, "cache_hits": evaluator.cache_hits,
            "tool_calls": budget.tool_calls, "max_tool_calls": config.max_tool_calls,
            "budget_utilization_percent": (100 * budget.tool_calls / config.max_tool_calls
                if budget.tool_calls is not None and config.max_tool_calls else
                0 if budget.tool_calls == 0 else None),
            "elapsed_seconds": time.perf_counter() - started, "status": status,
        }

    def emit(event, *, final=False):
        if progress is None:
            return
        state = {**snapshot(), "event": event, "status": status if final else "running"}
        if event == "configuration":
            state["config"] = asdict(config)
        if event == "checkpoint":
            state = checkpoints.checkpoint(state, final=final)
        if state is not None:
            progress(state)

    emit("configuration")
    try:
        baseline_result = abc.evaluate_recipe(
            circuit, [], budget=baseline_budget, metric="lut", lut_k=config.lut_k,
            timeout=config.timeout, abc_binary=config.abc_binary)
        baseline = valid_metrics(baseline_result.get("metrics"), baseline=True)
        baseline_evaluations = 1
        evaluator.baseline = baseline
        emit("baseline")
        # Iteration zero evaluates the initial population. Each later step uses
        # the PREVIOUS coefficients, exactly as upstream run().
        for iteration in range(config.generations + 1):
            if iteration:
                optimizer._step()
                optimizer._update_coefficients(iteration)
            population_evaluated = 0
            fitness = np.full(config.population_size, np.inf)
            for index, bits in enumerate(optimizer.pop):
                member = evaluator.evaluate(bits_to_recipe(bits))
                fitness[index] = member["weighted_fitness"]
                population_evaluated += 1
                if best is None or member["weighted_fitness"] < best["weighted_fitness"]:
                    best = member
                    emit("improvement")
                emit("checkpoint")
            optimizer._leaders.update(optimizer.pop, fitness, (1 - fitness) * 100)
            if iteration:
                completed_iterations = iteration
            emit("generation")
    except abc.BudgetExceeded as exc:
        if budget.tool_calls is None or baseline_budget.tool_calls is None:
            status, failure = "failed", {"type": type(exc).__name__, "message": str(exc)}
        else:
            status = "budget_exhausted"
    except (abc.ABCError, OSError, ValueError, RuntimeError) as exc:
        status = "failed"
        failure = {"type": type(exc).__name__, "message": str(exc),
                   "abc_output": getattr(exc, "output", ""),
                   "completed_actions_lower_bound": getattr(exc, "completed_actions", 0)}
    # KeyboardInterrupt deliberately propagates, like NSGA-II. The CLI cleared
    # old results before entry and the shared evaluator kills its active child.
    emit("checkpoint", final=True)
    emit("summary", final=True)
    executable = shutil.which(config.abc_binary)
    result = snapshot()
    result.update(
        schema_version=1, seed=config.seed, config=asdict(config), failure=failure,
        runtime_seconds=time.perf_counter() - started,
        best=best if status != "failed" else None,
        weighted_fitness=best["weighted_fitness"] if best and status != "failed" else None,
        provenance={
            "variant": "ast-gwo-binary-adaptive-sigmoid-v1",
            "implementation": "adapted from local ast-gwo-aicas2026; see algorithms/ast_gwo/PROVENANCE.md",
            "upstream_commit": "36686f1bc035d653da635c4748ecf8414e389a55",
            "numpy": np.__version__, "bit_dimension": 4 * config.recipe_length,
            "optimizer_sha256": fingerprint(Path(__file__).with_name("optimizer.py")),
            "python": platform.python_version(), "engine_sha256": fingerprint(__file__),
            "evaluator_sha256": fingerprint(abc.__file__), "action_pool": TRANSFORMS,
            "circuit_sha256": fingerprint(circuit) if Path(circuit).is_file() else None,
            "abc_executable": str(Path(executable).resolve()) if executable else None,
            "abc_sha256": fingerprint(executable) if executable else None,
        })
    return result
