"""Readable experiment identities and narrowly scoped serial rerun cleanup."""

import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import quote


REPOSITORY = Path(__file__).resolve().parents[1]
RUN_ARTIFACTS = ("config.json", "result.json", "best.csv", "best.json", "pareto.json", "progress.jsonl")
PARAMETERS = (
    ("population_size", "pop"), ("offspring_count", "off"),
    ("generations", "gen"), ("recipe_length", "len"),
)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None


def describe_experiment(algorithm, circuit, config):
    """Resolve input/tool identities without invoking ABC or changing config."""
    circuit = Path(circuit).resolve()
    try:
        identity = "repo/" + circuit.relative_to(REPOSITORY).as_posix()
    except ValueError:
        identity = "external/" + circuit.as_posix().lstrip("/")
    executable = shutil.which(config["abc_binary"])
    executable = str(Path(executable).resolve()) if executable else None
    return {
        "algorithm": algorithm, "circuit": str(circuit), "circuit_identity": identity,
        "circuit_sha256": fingerprint(circuit), "config": dict(config), "metric": "lut",
        "abc_executable": {"requested": config["abc_binary"], "resolved": executable,
                           "sha256": fingerprint(executable)},
    }


def component(value):
    """Reversible encoding of unsafe path characters."""
    if value is None:
        value = "none"
    elif isinstance(value, float):
        value = str(int(value)) if value.is_integer() else repr(value)
    value = quote(str(value), safe="")
    return value.replace(".", "%2E") if value in (".", "..") else value


def experiment_path(experiment, root="results"):
    """Readable paths using the declared varying parameters and seed."""
    config = experiment["config"]
    identity = experiment["circuit_identity"]
    parts = identity.split("/")
    if parts[:2] == ["repo", "benchmarks"]:
        suite = parts[3] if parts[2] == "converted" else parts[2]
    else:
        suite = Path(experiment["circuit"]).parent.name or "external"
    circuit = Path(experiment["circuit"]).stem
    tokens = [label + component(config[name]) for name, label in PARAMETERS if name in config]
    label = "_".join(tokens)
    return (Path(root) / component(experiment["algorithm"]) / component(suite)
            / component(circuit) / label / f"seed_{config['seed']}")


def ensure_directory(directory):
    """Refuse directory symlinks so rerun cleanup cannot follow them elsewhere."""
    directory = Path(directory)
    for parent in reversed((directory.absolute(), *directory.absolute().parents)):
        if parent.is_symlink():
            raise ValueError(f"Experiment directory must not traverse a symlink: {parent}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def clear_artifacts(directory, names=RUN_ARTIFACTS):
    directory = ensure_directory(directory)
    # Preflight all names; never recursively remove a directory or user files.
    for name in names:
        path = directory / name
        if path.exists() and not path.is_file() and not path.is_symlink():
            raise ValueError(f"Run artifact is not a file: {path}")
    for name in names:
        (directory / name).unlink(missing_ok=True)


def write_config(directory, experiment):
    (Path(directory) / "config.json").write_text(json.dumps(experiment, sort_keys=True, indent=2) + "\n")


def prepare_run(experiment, root="results"):
    """Clear only this experiment's known files before optimization starts."""
    directory = experiment_path(experiment, root)
    clear_artifacts(directory)
    write_config(directory, experiment)
    return directory
