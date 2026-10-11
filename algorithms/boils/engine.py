"""BOiLS baseline using the shared evaluator, scalar objective and artifacts."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import shutil
import time

import torch

from algorithms.greedy.engine import (Config as SharedConfig, RecipeEvaluator,
                                     valid_metrics, solution_key)
from algorithms.nsga2.engine import fingerprint, write_results
from common import abc
from common.actions import TRANSFORMS
from common.progress import CheckpointScheduler
from .optimizer import Optimizer


@dataclass(frozen=True)
class Config(SharedConfig):
    max_tool_calls: int = 200
    n_initial: int = 20
    ssk_order: int = 3
    gp_training_steps: int = 500
    restart_gp_training_steps: int = 400
    standardise: bool = True
    max_training_points: None = None
    candidate_pool_size: int | None = None
    acquisition_steps: int = 200
    tr_initial_radius: int | None = None
    tr_success_tolerance: int = 3
    tr_failure_tolerance: int = 40
    tr_multiplier: float = 1.5

    def __post_init__(self):
        if self.max_training_points is not None:
            raise ValueError("BOiLS uses full trust-region history; max_training_points must be null")
        if type(self.standardise) is not bool:
            raise ValueError("standardise must be a boolean")
        for name in ("candidate_pool_size", "tr_initial_radius"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be null or an integer >= 1")
        if self.optimization_mode != "weighted_sum":
            raise ValueError("BOiLS supports only optimization_mode weighted_sum")
        super().__post_init__()
        for name, minimum in (("n_initial", 1), ("ssk_order", 1), ("gp_training_steps", 1),
                              ("restart_gp_training_steps", 1), ("acquisition_steps", 0),
                              ("tr_success_tolerance", 1), ("tr_failure_tolerance", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.ssk_order > 3:
            raise ValueError("ssk_order must be <= 3")
        if type(self.tr_multiplier) not in (int, float) or not math.isfinite(self.tr_multiplier) or self.tr_multiplier <= 1:
            raise ValueError("tr_multiplier must be finite and > 1")


def load_config(path=None, **overrides):
    path = Path(path) if path else Path(__file__).with_name("config.yaml")
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object (YAML subset)")
    values.update({key: value for key, value in overrides.items() if value is not None})
    return Config(**values)


def run(circuit, config, *, progress=None):
    # Small dense CPU matrices run faster and reproducibly with one thread.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _run(circuit, config, progress=progress)
    finally:
        torch.set_num_threads(previous_threads)


def _run(circuit, config, *, progress=None):
    started = time.perf_counter()
    budget, baseline_budget = abc.ActionBudget(config.max_tool_calls), abc.ActionBudget(1)
    evaluator = RecipeEvaluator(circuit, config, None, budget)
    checkpoints = CheckpointScheduler(config.checkpoint_interval)
    baseline = best = None
    baseline_evaluations = iteration = 0
    optimizer = Optimizer(config)
    status, failure = "completed", None

    def snapshot():
        return {
            "algorithm": "boils", "circuit": str(circuit),
            "generation": iteration, "generations": config.max_tool_calls,
            "trust_region_radius": optimizer.region.radius,
            "restarts": optimizer.restarts, "surrogate_fits": optimizer.surrogate_fits,
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
        while True:
            budget.check()
            recipe = optimizer.suggest(evaluator.cache)
            if recipe is None:
                status = "search_space_exhausted"
                break
            cached = recipe in evaluator.cache
            member = evaluator.evaluate(recipe)
            iteration = evaluator.evaluations
            if best is None or solution_key(member) < solution_key(best):
                best = member
                emit("improvement")
            emit("checkpoint")
            if not cached:
                optimizer.observe(recipe, member["weighted_fitness"], evaluator.cache)
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
        observations=[{"recipe": list(r), "weighted_fitness": v} for r, v in optimizer.history],
        runtime_seconds=time.perf_counter() - started,
        best=best if status != "failed" else None,
        weighted_fitness=best["weighted_fitness"] if best and status != "failed" else None,
        provenance={
            "variant": "boils-upstream-settings-ssk-ei-cpu-v2",
            "implementation": "adapted from local HEBO/BOiLS; see algorithms/boils/PROVENANCE.md",
            "upstream_commit": "ee6112d39d1a9e9703fecaf9057193e1ec9dae72",
            "torch": torch.__version__, "device": "cpu", "dtype": "float64",
            "torch_threads": torch.get_num_threads(),
            "kernel": "ssk", "acquisition": "augmented_ei",
            "ard_requested_upstream": True, "ssk_has_ard": False,
            "initialization": "latin-hypercube-onehot-argmax",
            "effective_initial_radius": optimizer.initial_radius,
            "restart_candidate_count": optimizer.n_candidates,
            "optimizer_sha256": fingerprint(Path(__file__).with_name("optimizer.py")),
            "surrogate_sha256": fingerprint(Path(__file__).with_name("surrogate.py")),
            "python": platform.python_version(), "engine_sha256": fingerprint(__file__),
            "evaluator_sha256": fingerprint(abc.__file__), "action_pool": TRANSFORMS,
            "circuit_sha256": fingerprint(circuit) if Path(circuit).is_file() else None,
            "abc_executable": str(Path(executable).resolve()) if executable else None,
            "abc_sha256": fingerprint(executable) if executable else None,
        })
    return result
