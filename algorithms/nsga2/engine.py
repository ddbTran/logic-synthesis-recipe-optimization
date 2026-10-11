"""Independent discrete NSGA-II implementation; see PROVENANCE.md.

Only the shared evaluator executes ABC. Recipes are ordered tuples, not sets.
"""

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time

from common import abc
from common.actions import MAX_TOOL_CALLS, TRANSFORMS


VARIANT = "discrete-uniform-random-reset-nsga2-v1"
EXAMPLE_SHA256 = "b1a33ad832d067f8b21927980e1518ccdbad6d85840e7f376ad1b89346e797e3"


@dataclass(frozen=True)
class Config:
    seed: int = 0
    recipe_length: int = 20
    population_size: int = 40
    offspring_count: int = 40
    generations: int = 99
    crossover_rate: float = 0.9
    mutation_rate: float = 0.05
    lut_k: int = 6
    timeout: float | None = 60.0
    abc_binary: str = "yosys-abc"
    objective: str = "lut_count,lut_levels"
    max_tool_calls: int = MAX_TOOL_CALLS

    def __post_init__(self):
        for name, minimum in (("recipe_length", 1), ("population_size", 2),
                              ("offspring_count", 1), ("generations", 0),
                              ("lut_k", 2), ("max_tool_calls", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in ("crossover_rate", "mutation_rate"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.timeout is not None and (type(self.timeout) not in (int, float)
                                        or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("timeout must be null or finite and positive")
        if not isinstance(self.abc_binary, str) or not self.abc_binary:
            raise ValueError("abc_binary must be a nonempty string")
        if self.objective != "lut_count,lut_levels":
            raise ValueError("Only the two minimized objectives lut_count,lut_levels are supported")


def load_config(path=None, **overrides):
    # JSON is a YAML subset: deliberately no general YAML loader/dependency.
    path = Path(path) if path else Path(__file__).with_name("config.yaml")
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object (YAML subset)")
    values.update({key: value for key, value in overrides.items() if value is not None})
    return Config(**values)


@dataclass
class Individual:
    recipe: tuple[int, ...]
    objectives: tuple[int, int]
    rank: int = 0
    crowding: float = 0.0


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def nondominated_sort(population):
    """Deb's pairwise dominance bookkeeping; identity is by population index."""
    defeated = [[] for _ in population]
    counts = [0] * len(population)
    for i, a in enumerate(population):
        for j in range(i + 1, len(population)):
            b = population[j]
            if dominates(a.objectives, b.objectives):
                defeated[i].append(j)
                counts[j] += 1
            elif dominates(b.objectives, a.objectives):
                defeated[j].append(i)
                counts[i] += 1
    pending = [i for i, count in enumerate(counts) if count == 0]
    fronts = []
    while pending:
        front = []
        following = []
        for i in pending:
            population[i].rank = len(fronts)
            front.append(population[i])
            for j in defeated[i]:
                counts[j] -= 1
                if counts[j] == 0:
                    following.append(j)
        fronts.append(front)
        pending = following
    return fronts


def assign_crowding(front):
    for member in front:
        member.crowding = 0.0
    for objective in range(2):
        ordered = sorted(front, key=lambda p: p.objectives[objective])
        if not ordered:
            continue
        ordered[0].crowding = ordered[-1].crowding = math.inf
        span = ordered[-1].objectives[objective] - ordered[0].objectives[objective]
        if span:
            for left, current, right in zip(ordered, ordered[1:], ordered[2:]):
                current.crowding += (right.objectives[objective] - left.objectives[objective]) / span


def select_survivors(population, size):
    selected = []
    for front in nondominated_sort(population):
        assign_crowding(front)
        # Stable ordering resolves exact crowding ties reproducibly.
        selected.extend(sorted(front, key=lambda p: -p.crowding)[:size - len(selected)])
        if len(selected) == size:
            break
    # Recompute on survivors, as in the inspected discrete example.
    for front in nondominated_sort(selected):
        assign_crowding(front)
    return selected


def tournament(population, rng):
    a, b = rng.sample(population, 2)
    return min((a, b), key=lambda p: (p.rank, -p.crowding))


def vary(a, b, rng, config):
    children = [list(a), list(b)]
    if rng.random() <= config.crossover_rate:
        for i in range(len(a)):
            if rng.random() >= 0.5:
                children[0][i], children[1][i] = children[1][i], children[0][i]
    return [tuple(rng.randrange(len(TRANSFORMS)) if rng.random() < config.mutation_rate
                  else gene for gene in child) for child in children]


def solution(member):
    return {"recipe": list(member.recipe), "lut_count": member.objectives[0],
            "lut_levels": member.objectives[1]}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(circuit, config, *, progress=None):
    started = time.perf_counter()
    budget = abc.ActionBudget(config.max_tool_calls)
    rng = random.Random(config.seed)
    cache = {}
    archive = {}
    evaluations = cache_hits = completed_generations = 0
    status = "completed"
    failure = None

    def emit(event, generation, event_status="running"):
        if progress is not None:
            ordered = sorted(archive.values(), key=lambda p: (*p.objectives, p.recipe))
            calls = budget.tool_calls
            progress({
                "event": event, "generation": generation, "generations": config.generations,
                "best": solution(ordered[0]) if ordered else None,
                "pareto_size": len(archive), "tool_calls": calls,
                "max_tool_calls": config.max_tool_calls,
                "budget_utilization_percent": (100 * calls / config.max_tool_calls
                                               if calls is not None and config.max_tool_calls else
                                               0 if calls == 0 else None),
                "cache_hits": cache_hits, "elapsed_seconds": time.perf_counter() - started,
                "status": event_status,
                **({"config": asdict(config)} if event == "configuration" else {}),
            })

    generation = 0
    emit("configuration", generation)

    def evaluate(recipe):
        nonlocal evaluations, cache_hits
        if recipe in cache:
            cache_hits += 1
            return Individual(recipe, cache[recipe])
        budget.check()
        result = abc.evaluate_recipe(circuit, recipe, budget=budget, metric="lut",
                                     lut_k=config.lut_k, timeout=config.timeout,
                                     abc_binary=config.abc_binary)
        if budget.tool_calls is None:
            raise abc.ABCError("Evaluator returned with uncertain budget")
        evaluations += 1
        metrics = result["metrics"]
        objectives = (metrics["lut_count"], metrics["lut_levels"])
        cache[recipe] = objectives
        member = Individual(recipe, objectives)
        previous_best = min(((*p.objectives, p.recipe) for p in archive.values()), default=None)
        if not any(dominates(p.objectives, objectives) for p in archive.values()):
            for old in list(archive):
                if dominates(objectives, archive[old].objectives):
                    del archive[old]
            archive[recipe] = member
        if previous_best is None or (*objectives, recipe) < previous_best:
            emit("improvement", generation)
        return member

    try:
        population = [evaluate(tuple(rng.randrange(len(TRANSFORMS))
                                     for _ in range(config.recipe_length)))
                      for _ in range(config.population_size)]
        population = select_survivors(population, config.population_size)
        emit("generation", 0)
        for generation in range(1, config.generations + 1):
            offspring = []
            while len(offspring) < config.offspring_count:
                a, b = tournament(population, rng), tournament(population, rng)
                for recipe in vary(a.recipe, b.recipe, rng, config):
                    if len(offspring) == config.offspring_count:
                        break
                    offspring.append(evaluate(recipe))
            population = select_survivors(population + offspring, config.population_size)
            completed_generations += 1
            emit("generation", generation)
    except abc.BudgetExceeded as exc:
        if budget.tool_calls is None:
            status, failure = "failed", {"type": type(exc).__name__, "message": str(exc)}
        else:
            status = "budget_exhausted"
    except (abc.ABCError, OSError, ValueError) as exc:
        status = "failed"
        failure = {"type": type(exc).__name__, "message": str(exc),
                   "abc_output": getattr(exc, "output", ""),
                   "completed_actions_lower_bound": getattr(exc, "completed_actions", 0)}
    ordered = sorted(archive.values(), key=lambda p: (*p.objectives, p.recipe))
    emit("summary", generation, status)
    executable = shutil.which(config.abc_binary)
    provenance = {
        "variant": VARIANT, "source_example_sha256": EXAMPLE_SHA256,
        "implementation": "independent; no external source copied",
        "python": platform.python_version(), "engine_sha256": fingerprint(__file__),
        "evaluator_sha256": fingerprint(abc.__file__),
        "action_pool": TRANSFORMS,
        "circuit_sha256": fingerprint(circuit) if Path(circuit).is_file() else None,
        "abc_executable": str(Path(executable).resolve()) if executable else None,
        "abc_sha256": fingerprint(executable) if executable else None,
    }
    return {
        "schema_version": 1, "algorithm": "nsga2", "circuit": str(circuit),
        "seed": config.seed, "status": status,
        "best": solution(ordered[0]) if ordered and status != "failed" else None,
        "pareto": [solution(p) for p in ordered],
        "tool_calls": budget.tool_calls, "runtime_seconds": time.perf_counter() - started,
        "successful_evaluations": evaluations, "cache_hits": cache_hits,
        "completed_generations": completed_generations,
        "config": asdict(config), "provenance": provenance, "failure": failure,
    }


def write_results(result, root="results", *, directory=None):
    """Unique run directory prevents stale success rows after a failed rerun."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(directory) if directory is not None else Path(tempfile.mkdtemp(prefix="nsga2-", dir=root))
    (directory / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    (directory / "pareto.json").write_text(json.dumps(result["pareto"], sort_keys=True) + "\n")
    if result["status"] != "failed" and result["best"] is not None and result["tool_calls"] is not None:
        best = result["best"]
        row = {key: result[key] for key in ("algorithm", "circuit", "seed", "status", "tool_calls", "runtime_seconds")}
        row.update(best_recipe=json.dumps(best["recipe"]), metric="lut", lut_k=result["config"]["lut_k"],
                   lut_count=best["lut_count"], lut_levels=best["lut_levels"],
                   config=json.dumps(result["config"], sort_keys=True),
                   provenance=json.dumps(result["provenance"], sort_keys=True))
        with (directory / "best.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    return directory
