"""Original DRiLLS pre-mapping Verilog statistics; see PROVENANCE.md.

Adapted from SCALE Lab, Brown University (2019), BSD-3-Clause; see LICENSE.
No synthesis transformations or reward calculations live in this module.
"""

import math
import re
import shutil
import subprocess

from common.abc import ABCError

FEATURE_NAMES = (
    "primary_inputs", "primary_outputs", "cells", "edges", "logic_levels",
    "latches", "and_fraction", "or_fraction", "not_fraction",
)


def validate_features(features):
    if not isinstance(features, (list, tuple)) or len(features) != len(FEATURE_NAMES):
        raise ValueError("DRiLLS requires exactly nine observation features")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in features):
        raise ValueError("DRiLLS features must be finite and nonnegative")
    if any(x > 1 for x in features[6:]):
        raise ValueError("Gate fractions must be within [0, 1]")
    return list(features)


def parse_features(yosys_output, abc_output):
    """Use Yosys read_verilog;stat cells and ABC read_verilog;print_stats."""
    cells = re.findall(r"Number of cells:\s*(\d+)", yosys_output)
    if len(cells) != 1:
        raise ABCError("Expected one Yosys module with cell statistics", yosys_output)
    cells = int(cells[0])
    gates = []
    for gate in ("and", "or", "not"):
        matches = re.findall(rf"(?m)^\s*\${gate}\s+(\d+)\s*$", yosys_output)
        count = int(matches[0]) if matches else 0
        if count > cells:
            raise ABCError("Invalid Yosys gate count", yosys_output)
        gates.append(count / cells if cells else 0.0)
    clean = re.sub(r"\x1b\[[0-9;]*m", "", abc_output)
    for line in reversed(clean.splitlines()):
        pins = re.search(r"\bi/o\s*=\s*(\d+)\s*/\s*(\d+)", line)
        values = [re.search(rf"\b{name}\s*=\s*(\d+)\b", line)
                  for name in ("edge", "lev", "lat")]
        if pins and all(values):
            edges, levels, latches = [int(match[1]) for match in values]
            return validate_features([int(pins[1]), int(pins[2]), cells,
                                      edges, levels, latches, *gates])
    raise ABCError("Missing original DRiLLS ABC feature statistics", abc_output)


class FeatureExtractor:
    """Two auxiliary read-only tool processes per uncached observation."""

    def __init__(self, abc_binary="yosys-abc", yosys_binary="yosys", timeout=60):
        self.abc_binary, self.yosys_binary, self.timeout = abc_binary, yosys_binary, timeout
        self.abc_process_invocations = self.yosys_process_invocations = 0
        self.failed_processes = self.timed_out_processes = 0
        self.launch_uncertain = False

    def _run(self, binary, arguments, directory, kind):
        executable = shutil.which(binary)
        if executable is None:
            raise ABCError(f"Feature executable unavailable: {binary}")
        try:
            process = subprocess.Popen([executable, *arguments], cwd=directory,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except OSError as exc:
            raise ABCError(f"Cannot launch feature extractor: {exc}") from exc
        except BaseException:
            self.launch_uncertain = True
            raise
        name = f"{kind}_process_invocations"
        setattr(self, name, getattr(self, name) + 1)
        try:
            output, _ = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            output, _ = process.communicate()
            self.failed_processes += 1
            self.timed_out_processes += 1
            raise ABCError(f"Feature {kind} timed out after {self.timeout}s", output) from exc
        except BaseException:
            process.kill()
            process.wait()
            self.failed_processes += 1
            raise
        if process.returncode or re.search(r"(?im)^\s*(?:ERROR\b|\*\*\s*cmd error|unknown command\b|usage:)", output):
            self.failed_processes += 1
            raise ABCError(f"Feature {kind} failed (exit {process.returncode})", output)
        return output

    def __call__(self, path):
        # Shared evaluator supplies this fixed safe local export name, still live.
        if path.name != "observation.v" or not path.is_file():
            raise ABCError("Missing pre-mapping observation.v export")
        yosys = self._run(self.yosys_binary,
            ["-QT", "-p", "read_verilog observation.v; stat"], path.parent, "yosys")
        abc = self._run(self.abc_binary,
            ["-s", "-c", "read_verilog observation.v; print_stats"], path.parent, "abc")
        return parse_features(yosys, abc)

    def counters(self):
        return {
            "auxiliary_abc_process_invocations": self.abc_process_invocations,
            "auxiliary_yosys_process_invocations": self.yosys_process_invocations,
            "auxiliary_process_invocations": self.abc_process_invocations + self.yosys_process_invocations,
            "failed_auxiliary_processes": self.failed_processes,
            "timed_out_auxiliary_processes": self.timed_out_processes,
            "auxiliary_launch_uncertain": self.launch_uncertain,
        }
