"""DRiLLS A2C adaptation using shared replay evaluation and weighted QoR reward."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import re
import shutil
import time

from algorithms.greedy.engine import Config as EvaluationConfig
from algorithms.nsga2.engine import write_results
from common import abc
from common.actions import TRANSFORMS
from common.experiments import fingerprint
from common.progress import CheckpointScheduler
from .environment import Environment
from .features import FEATURE_NAMES, FeatureExtractor

SOURCE_REVISION = "c8350ce3ac4198733b523cb52d5ba79aec0ed8aa"


@dataclass(frozen=True)
class Config(EvaluationConfig):
    recipe_length: int = 20
    episodes: int = 200
    gamma: float = 0.99
    actor_learning_rate: float = 0.01
    critic_learning_rate: float = 0.01
    actor_hidden: tuple = (20, 20)
    critic_hidden: tuple = (10,)
    yosys_binary: str = "yosys"
    device: str = "auto"
    deterministic: bool = False
    objective: str = "lut_count,lut_levels"

    def __post_init__(self):
        super().__post_init__()
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic must be boolean")
        if type(self.episodes) is not int or self.episodes < 1:
            raise ValueError("episodes must be an integer >= 1")
        if not 0 <= self.seed < 2**32:
            raise ValueError("DRiLLS seed must be within [0, 2**32)")
        for name in ("gamma", "actor_learning_rate", "critic_learning_rate"):
            value = getattr(self, name)
            minimum = 0.0
            if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
                raise ValueError(f"{name} must be finite and nonnegative")
            if "learning_rate" in name and value == 0:
                raise ValueError(f"{name} must be positive")
        if self.gamma > 1:
            raise ValueError("gamma must be within [0, 1]")
        for name, count in (("actor_hidden", 2), ("critic_hidden", 1)):
            widths = getattr(self, name)
            if not isinstance(widths, (list, tuple)) or len(widths) != count or any(
                type(w) is not int or w < 1 for w in widths
            ):
                raise ValueError(f"{name} requires {count} positive integer widths")
            object.__setattr__(self, name, tuple(widths))
        if not isinstance(self.yosys_binary, str) or not self.yosys_binary:
            raise ValueError("yosys_binary must be a nonempty string")
        if not isinstance(self.device, str) or not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", self.device):
            raise ValueError("device must be auto, cpu, cuda or cuda:N")
        if self.objective != "lut_count,lut_levels":
            raise ValueError("Only objective lut_count,lut_levels is supported")


def load_config(path=None, **overrides):
    path = Path(path) if path else Path(__file__).with_name("config.yaml")
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object (YAML subset)")
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**values)


def dependencies():
    try:
        from . import model
    except ImportError as exc:
        raise ValueError("DRiLLS requires PyTorch and NumPy; install root requirements.txt with Python 3.11") from exc
    return model


def tool_identity(binary):
    resolved = shutil.which(binary)
    return {"requested": binary, "resolved": str(Path(resolved).resolve()) if resolved else None,
            "sha256": fingerprint(resolved)}


def run(circuit, config, *, progress=None):
    model = dependencies()
    model.seed_all(config.seed, config.deterministic)
    started = time.perf_counter()
    budget, baseline_budget = abc.ActionBudget(config.max_tool_calls), abc.ActionBudget(1)
    extractor = FeatureExtractor(config.abc_binary, config.yosys_binary, config.timeout)
    env = Environment(circuit, config, budget, extractor)
    scheduler = CheckpointScheduler(config.checkpoint_interval)
    episode = depth = completed_episodes = incomplete_episodes = baseline_evaluations = 0
    baseline_failures = baseline_timeouts = 0
    history = []
    agent = None
    states, actions, rewards = [], [], []
    status, failure = "completed", None

    def snapshot():
        best = env.best
        return {
            "algorithm": "drills", "circuit": str(circuit), "episode": episode,
            "episodes": config.episodes, "depth": depth, "target_depth": config.recipe_length,
            "optimization_mode": config.optimization_mode,
            "weight_area": config.weight_area, "weight_delay": config.weight_delay,
            "baseline_qor": env.baseline, "baseline_observation": env.baseline_features,
            "baseline_evaluations": baseline_evaluations, "baseline_tool_calls": baseline_budget.tool_calls,
            "failed_baseline_evaluations": baseline_failures, "timed_out_baseline_evaluations": baseline_timeouts,
            "best": best, "best_so_far": best, "best_complete": env.best_complete,
            "weighted_fitness": best["weighted_fitness"] if best else None,
            "best_fitness": best["weighted_fitness"] if best else None,
            "best_lut_count": best["lut_count"] if best else None,
            "best_lut_levels": best["lut_levels"] if best else None,
            "evaluations": env.evaluations, "successful_evaluations": env.evaluations,
            "candidate_attempts": env.candidate_attempts, "cache_hits": env.cache_hits,
            "failed_evaluations": env.failed_evaluations, "timed_out_evaluations": env.timed_out_evaluations,
            "tool_calls": budget.tool_calls, "max_tool_calls": config.max_tool_calls,
            "abc_process_invocations": budget.abc_process_invocations + baseline_budget.abc_process_invocations,
            "total_abc_process_invocations": (budget.abc_process_invocations + baseline_budget.abc_process_invocations
                                              + extractor.abc_process_invocations),
            "transformation_executions": env.transformation_executions,
            "confirmed_failed_transformations": env.confirmed_failed_transformations,
            **extractor.counters(),
            "completed_episodes": completed_episodes, "incomplete_episodes": incomplete_episodes,
            "training_updates": agent.updates if agent else 0,
            "requested_device": config.device,
            "resolved_device": str(agent.device) if agent else None,
            "torch_version": model.torch.__version__,
            "cuda_available": model.torch.cuda.is_available(),
            "cuda_runtime_version": model.torch.version.cuda,
            "gpu_model": model.torch.cuda.get_device_name(agent.device)
                         if agent and agent.device.type == "cuda" else None,
            "budget_utilization_percent": (100 * budget.tool_calls / config.max_tool_calls
                if budget.tool_calls is not None and config.max_tool_calls else 0 if budget.tool_calls == 0 else None),
            "elapsed_seconds": time.perf_counter() - started, "status": status,
        }

    def emit(event, *, final=False, **extra):
        if progress is None:
            return
        state = {**snapshot(), "event": event, "status": status if final else "running", **extra}
        if event == "configuration":
            state["config"] = asdict(config)
        if event == "checkpoint":
            state = scheduler.checkpoint(state, final=final)
        if state is not None:
            progress(state)

    emit("configuration")
    try:
        agent = model.Agent(config)
        try:
            env.initialize(baseline_budget)
            baseline_evaluations = 1
        except (abc.ABCError, OSError, ValueError) as exc:
            baseline_failures = 1
            baseline_timeouts = int("timed out" in str(exc).lower())
            raise
        emit("baseline")
        for episode in range(1, config.episodes + 1):
            depth = 0
            states, actions, rewards = [], [], []
            normalizer = model.Normalizer()
            state = normalizer.observe_and_normalize(env.reset())
            for depth in range(1, config.recipe_length + 1):
                action = agent.sample(state)
                member, reward, done, improved = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                if improved:
                    emit("improvement")
                emit("checkpoint")
                state = normalizer.observe_and_normalize(member["observation"])
                if done:
                    break
            # Complete trajectories only; zero terminal bootstrap as in source.
            training = agent.update(states, actions, rewards)
            completed_episodes += 1
            history.append({"episode": episode, "length": len(rewards), "recipe": list(env.recipe),
                            "rewards": rewards, "reward": sum(rewards), "complete": True,
                            "trained": True, **training})
            emit("episode", training=history[-1])
    except abc.BudgetExceeded as exc:
        if budget.tool_calls is None or baseline_budget.tool_calls is None:
            status, failure = "failed", {"type": type(exc).__name__, "message": str(exc)}
        else:
            status = "budget_exhausted"
    except (abc.ABCError, OSError, ValueError, RuntimeError, OverflowError) as exc:
        status, failure = "failed", {"type": type(exc).__name__, "message": str(exc),
            "abc_output": getattr(exc, "output", ""),
            "completed_actions_lower_bound": getattr(exc, "completed_actions", 0)}
    # KeyboardInterrupt propagates; shared runner pre-clears artifacts and the
    # evaluator/extractor kills its children. There is no training-resume claim.
    if episode > completed_episodes:
        incomplete_episodes = 1
        history.append({"episode": episode, "length": len(rewards), "recipe": list(env.recipe),
                        "rewards": rewards, "reward": sum(rewards), "complete": False,
                        "trained": False, "termination_status": status})
    emit("checkpoint", final=True)
    emit("summary", final=True)
    result = snapshot()
    abc_identity, yosys_identity = tool_identity(config.abc_binary), tool_identity(config.yosys_binary)
    result.update(schema_version=1, seed=config.seed, config=asdict(config), failure=failure,
        runtime_seconds=time.perf_counter()-started, episode_history=history,
        best=env.best if status != "failed" else None,
        weighted_fitness=env.best["weighted_fitness"] if env.best and status != "failed" else None,
        provenance={
            "variant": "drills-source-a2c-weighted-improvement-v1", "source_revision": SOURCE_REVISION,
            "source_repository": "https://github.com/scale-lab/DRiLLS", "python": platform.python_version(),
            "torch": model.torch.__version__, "numpy": model.np.__version__,
            "requested_device": config.device, "resolved_device": str(agent.device) if agent else None,
            "cuda_available": model.torch.cuda.is_available(), "torch_cuda_version": model.torch.version.cuda,
            "gpu_model": result["gpu_model"], "deterministic": config.deterministic,
            "architecture": {"actor": [9, *config.actor_hidden, len(TRANSFORMS)],
                             "critic": [9, *config.critic_hidden, 1]},
            "feature_names": FEATURE_NAMES, "feature_normalization": "episode-reset Welford; variance clamp [0.01, 1e9]",
            "loss_semantics": "literal source [T,T] broadcast; summed losses; actor gradients into critic",
            "partial_episode_policy": "skip training; preserve valid evaluations", "resume_supported": False,
            "action_pool": TRANSFORMS, "circuit_sha256": fingerprint(circuit),
            "abc_executable": abc_identity["resolved"], "abc_sha256": abc_identity["sha256"],
            "yosys_executable": yosys_identity,
            "source_hashes": {name: fingerprint(Path(__file__).with_name(name))
                              for name in ("engine.py", "model.py", "features.py", "environment.py")},
            "evaluator_sha256": fingerprint(abc.__file__),
        })
    return result
