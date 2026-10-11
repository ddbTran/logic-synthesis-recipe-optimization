"""Sequential Forward Greedy over the shared ordered ABC action pool.

Independent algorithm, not a reproduction of a paper's Greedy baseline.
The scalar objective and artifact writer are reused from the reference engine.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import shutil
import time

from algorithms.nsga2.engine import fingerprint, weighted_fitness, write_results
from common import abc
from common.actions import MAX_TOOL_CALLS, TRANSFORMS
from common.progress import CheckpointScheduler


@dataclass(frozen=True)
class Config:
    recipe_length: int = 20
    seed: int = 0
    max_tool_calls: int = MAX_TOOL_CALLS
    checkpoint_interval: int = 200
    optimization_mode: str = "weighted_sum"
    lut_k: int = 6
    weight_area: float = 0.5
    weight_delay: float = 0.5
    abc_binary: str = "yosys-abc"
    timeout: float | None = 60.0

    def __post_init__(self):
        for name, minimum in (("recipe_length", 1), ("max_tool_calls", 0),
                              ("checkpoint_interval", 1), ("lut_k", 2)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.optimization_mode != "weighted_sum":
            raise ValueError("Greedy supports only optimization_mode weighted_sum")
        for name in ("weight_area", "weight_delay"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isclose(self.weight_area + self.weight_delay, 1, rel_tol=0, abs_tol=1e-9):
            raise ValueError("weight_area + weight_delay must equal 1 within 1e-9")
        if not isinstance(self.abc_binary, str) or not self.abc_binary:
            raise ValueError("abc_binary must be a nonempty string")
        if self.timeout is not None and (type(self.timeout) not in (int, float)
                                        or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("timeout must be null or finite and positive")


def load_config(path=None, **overrides):
    path = Path(path) if path else Path(__file__).with_name("config.yaml")
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object (YAML subset)")
    values.update({key: value for key, value in overrides.items() if value is not None})
    return Config(**values)


def valid_metrics(metrics, *, baseline=False):
    """Require measured integer QoR; normalization additionally needs positivity."""
    if not isinstance(metrics, dict) or any(
        type(metrics.get(name)) is not int or metrics[name] < (1 if baseline else 0)
        for name in ("lut_count", "lut_levels")
    ):
        raise ValueError("Baseline QoR must be positive integers" if baseline
                         else "Candidate QoR must be nonnegative integers")
    return {name: metrics[name] for name in ("lut_count", "lut_levels")}


class RecipeEvaluator:
    """Run-local ordered-recipe cache in a fixed circuit/config/baseline context."""

    def __init__(self, circuit, config, baseline, budget):
        self.circuit, self.config, self.baseline, self.budget = circuit, config, baseline, budget
        self.cache = {}
        self.candidate_attempts = self.cache_hits = self.evaluations = 0

    def evaluate(self, recipe):
        recipe = tuple(recipe)
        self.candidate_attempts += 1
        if recipe in self.cache:
            self.cache_hits += 1
            return dict(self.cache[recipe], recipe=list(recipe))
        self.budget.check()
        result = abc.evaluate_recipe(
            self.circuit, recipe, budget=self.budget, metric="lut", lut_k=self.config.lut_k,
            timeout=self.config.timeout, abc_binary=self.config.abc_binary)
        if self.budget.tool_calls is None:
            raise abc.ABCError("Evaluator returned with uncertain budget")
        metrics = valid_metrics(result.get("metrics"))
        fitness = weighted_fitness((metrics["lut_count"], metrics["lut_levels"]),
                                   self.baseline, self.config)
        if not math.isfinite(fitness):
            raise ValueError("Candidate weighted fitness must be finite")
        member = {"recipe": list(recipe), **metrics, "weighted_fitness": fitness}
        self.cache[recipe] = member
        self.evaluations += 1
        return dict(member, recipe=list(recipe))


def solution_key(member):
    # At a fixed depth all recipes share a prefix, so this ties by action index.
    return member["weighted_fitness"], tuple(member["recipe"])


def run(circuit, config, *, progress=None):
    started = time.perf_counter()
    budget, baseline_budget = abc.ActionBudget(config.max_tool_calls), abc.ActionBudget(1)
    evaluator = RecipeEvaluator(circuit, config, None, budget)
    checkpoints = CheckpointScheduler(config.checkpoint_interval)
    baseline = best = best_complete = constructed = None
    baseline_evaluations = completed_depth = depth = depth_candidates = 0
    prefix = ()
    status, failure = "completed", None

    def snapshot():
        return {
            "algorithm": "greedy", "circuit": str(circuit),
            "depth": depth, "target_depth": config.recipe_length,
            "completed_depth": completed_depth, "depth_candidates_evaluated": depth_candidates,
            "optimization_mode": config.optimization_mode,
            "weight_area": config.weight_area, "weight_delay": config.weight_delay,
            "baseline_qor": baseline, "baseline_evaluations": baseline_evaluations,
            "baseline_tool_calls": baseline_budget.tool_calls,
            "abc_process_invocations": budget.abc_process_invocations + baseline_budget.abc_process_invocations,
            "best": best, "best_so_far": best, "best_complete": best_complete,
            "constructed_recipe": list(prefix), "constructed_solution": constructed,
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
        for depth in range(1, config.recipe_length + 1):
            depth_candidates = 0
            winner = None
            for action in range(len(TRANSFORMS)):
                member = evaluator.evaluate(prefix + (action,))
                depth_candidates += 1
                if winner is None or solution_key(member) < solution_key(winner):
                    winner = member
                if depth == config.recipe_length and (
                    best_complete is None or solution_key(member) < solution_key(best_complete)
                ):
                    best_complete = member
                if best is None or solution_key(member) < solution_key(best):
                    best = member
                    emit("improvement")
                emit("checkpoint")
            # Commit only a selection that compared the entire action pool.
            constructed = winner
            prefix = tuple(winner["recipe"])
            completed_depth = depth
            emit("step")
    except abc.BudgetExceeded as exc:
        if budget.tool_calls is None or baseline_budget.tool_calls is None:
            status, failure = "failed", {"type": type(exc).__name__, "message": str(exc)}
        else:
            status = "budget_exhausted"
    except (abc.ABCError, OSError, ValueError) as exc:
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
            "variant": "sequential-forward-greedy-v1",
            "implementation": "independent Sequential Forward Greedy; no external source copied",
            "python": platform.python_version(), "engine_sha256": fingerprint(__file__),
            "evaluator_sha256": fingerprint(abc.__file__), "action_pool": TRANSFORMS,
            "circuit_sha256": fingerprint(circuit) if Path(circuit).is_file() else None,
            "abc_executable": str(Path(executable).resolve()) if executable else None,
            "abc_sha256": fingerprint(executable) if executable else None,
        })
    return result
