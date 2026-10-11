"""Serial ABC recipe evaluation; no algorithm, reward, or ASIC mapping policy."""

from dataclasses import dataclass
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid

from common.actions import TRANSFORMS


class ABCError(RuntimeError):
    """Output and diagnostic action completions survive failed evaluations."""

    def __init__(self, message, output="", completed_actions=0):
        super().__init__(message)
        self.output = output
        # Lower bound only on failures; the interrupted action may have run.
        self.completed_actions = completed_actions


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ActionBudget:
    """ABC pool-action budget. One per run, serial use only."""

    max_tool_calls: int
    tool_calls: int | None = 0
    abc_process_invocations: int = 0

    def __post_init__(self):
        if type(self.max_tool_calls) is not int or self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be a nonnegative integer")
        if type(self.tool_calls) is not int or not 0 <= self.tool_calls <= self.max_tool_calls:
            raise ValueError("initial tool_calls must be within the budget")

    def check(self, count=1):
        if self.tool_calls is None:
            raise BudgetExceeded("Previous ABC launch left action count uncertain; stop this run")
        if self.tool_calls + count > self.max_tool_calls:
            raise BudgetExceeded(f"Evaluation needs {count} tool calls; {self.max_tool_calls - self.tool_calls} remain")


def parse_qor(output, metric):
    """Parse the last print_stats line for the explicitly chosen network type."""
    if metric not in ("aig", "lut"):
        raise ValueError("metric must be 'aig' or 'lut'; ASIC evaluation is not implemented")
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    node_label = "and" if metric == "aig" else "nd"
    for line in reversed(output.splitlines()):
        nodes = re.search(rf"\b{node_label}\s*=\s*(\d+)\b", line)
        levels = re.search(r"\blev\s*=\s*(\d+)\b", line)
        if nodes and levels:
            return {
                "aig_nodes" if metric == "aig" else "lut_count": int(nodes[1]),
                "aig_levels" if metric == "aig" else "lut_levels": int(levels[1]),
            }
    raise ABCError(f"Missing {metric} print_stats metrics", output)


def evaluate_recipe(circuit, recipe, *, budget, metric, abc_binary="yosys-abc", lut_k=None, timeout=60.0):
    """Reload circuit, strash, apply indices, and measure AIG or explicitly K-LUT QoR.

    Each executed pool entry costs one action call (including a GIA bundle).
    Reserve the whole recipe before launch; loading/mapping cost zero actions.
    Rejected inputs and definite launch failures cost zero. No cache here.
    Failed runs retain an exact count only if all actions have completion markers;
    otherwise the interrupted action's execution is unknown and accounting is null.
    timeout=None disables the subprocess deadline. Action markers are diagnostic.
    An unexpected launch exception makes accounting uncertain and stops the run.
    """
    recipe = list(recipe)
    if any(type(index) is not int or not 0 <= index < len(TRANSFORMS) for index in recipe):
        raise ValueError("Recipe must contain integer action indices 0-15")
    if metric not in ("aig", "lut"):
        raise ValueError("metric must be 'aig' or 'lut'; ASIC evaluation is not implemented")
    if metric == "lut" and (type(lut_k) is not int or lut_k < 2):
        raise ValueError("LUT evaluation requires an explicit integer lut_k >= 2 supported by ABC")
    if metric == "aig" and lut_k is not None:
        raise ValueError("lut_k applies only to LUT evaluation")
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be None or finite and positive")
    circuit = Path(circuit).resolve(strict=True)
    if not circuit.is_file():
        raise ValueError("Circuit must be a file")
    if circuit.suffix.lower() not in (".aig", ".aiger", ".blif", ".bench", ".v"):
        raise ValueError("Use an ABC-readable .aig, .aiger, .blif, .bench, or .v circuit")
    executable = shutil.which(str(abc_binary))
    if executable is None:
        raise ABCError(f"ABC executable unavailable: {abc_binary}")
    executable = str(Path(executable).resolve())
    budget.check(len(recipe))
    previous_calls = budget.tool_calls
    token = uuid.uuid4().hex
    action_prefix = f"ACTION_{token}_"
    finished = f"FINISHED_{token}"
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="recipe-abc-") as directory:
        directory = Path(directory)
        # A safe local name avoids interpolating caller paths into ABC commands.
        local_circuit = directory / ("input" + circuit.suffix)
        local_circuit.symlink_to(circuit)
        commands = [f"read {local_circuit.name}", "strash"]
        for position, index in enumerate(recipe):
            commands.extend([TRANSFORMS[index], f"echo {action_prefix}{position}"])
        if metric == "lut":
            commands.append(f"if -K {lut_k}")
        commands.extend(["print_stats", f"echo {finished}"])
        script = directory / "recipe.abc"
        script.write_text(";\n".join(commands) + ";\n")
        try:
            # -s disables user aliases; -f avoids echoing marker-containing commands.
            process = subprocess.Popen(
                [executable, "-s", "-f", script.name], cwd=directory,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
        except OSError as exc:
            # Popen reports failure to create/exec the child: no ABC evaluation.
            raise ABCError(f"Cannot execute ABC: {exc}") from exc
        except Exception as exc:
            budget.tool_calls = None
            raise ABCError(f"ABC launch state uncertain; stop this run: {exc}") from exc
        except BaseException:
            # No process handle: cannot establish whether ABC started. Fail closed.
            budget.tool_calls = None
            raise
        # Reserve the maximum action cost; no next recipe can exceed the limit.
        budget.tool_calls += len(recipe)
        budget.abc_process_invocations += 1
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            output, _ = process.communicate()
            lines = {line.strip() for line in output.splitlines()}
            confirmed = sum(f"{action_prefix}{i}" in lines for i in range(len(recipe)))
            budget.tool_calls = previous_calls + confirmed if confirmed == len(recipe) else None
            raise ABCError(f"ABC timed out after {timeout}s; stop this run", output, confirmed) from exc
        except Exception as exc:
            process.kill()
            process.wait()
            budget.tool_calls = previous_calls if not recipe else None
            raise ABCError(f"Cannot collect ABC evaluation output: {exc}") from exc
        except BaseException:
            # Ensure no evaluation survives; interrupted action accounting is unknown.
            process.kill()
            process.wait()
            budget.tool_calls = previous_calls if not recipe else None
            raise
    lines = [line.strip() for line in output.splitlines()]
    confirmed = sum(f"{action_prefix}{i}" in lines for i in range(len(recipe)))
    budget.tool_calls = previous_calls + confirmed if confirmed == len(recipe) else None
    # ABC may report command errors while returning exit status zero.
    errors = re.search(r"(?im)^\s*(?:\*\*\s*cmd error|error\b|unknown command\b|usage:)", output)
    if process.returncode or errors or finished not in lines or confirmed != len(recipe):
        raise ABCError(f"ABC evaluation failed (exit {process.returncode}); inspect output", output, confirmed)
    try:
        metrics = parse_qor(output, metric)
    except ABCError as exc:
        exc.completed_actions = confirmed
        raise
    return {
        "recipe": recipe,
        "metric": metric,
        "lut_k": lut_k,
        "metrics": metrics,
        "tool_calls": confirmed,
        "transformation_count": confirmed,
        "total_tool_calls": budget.tool_calls,
        "abc_process_invocations": 1,
        "full_recipe_evaluations": 1,
        "runtime_seconds": time.perf_counter() - started,
        "abc_output": output,
    }
