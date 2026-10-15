#!/usr/bin/env bash
set -uo pipefail

# Usage: ./run.sh {nsga2|greedy|drills|boils|ast_gwo}
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {nsga2|greedy|drills|boils|ast_gwo}" >&2
  exit 2
fi
case "$1" in
  nsga2|greedy|drills|boils|ast_gwo) algorithm="$1" ;;
  *) echo "Unsupported algorithm: $1" >&2; exit 2 ;;
esac

SEEDS=(0 1 2 3 4)
MAX_TOOL_CALLS=4000
LUT_K=6
WEIGHT_AREA=0.5
WEIGHT_DELAY=0.5
SRUN_ARGS=(--time=7-00:00:00 --mem=4G --cpus-per-task=2)

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" || exit 1
[[ -f run.py ]] || { echo 'Missing run.py; place run.sh at the repository root' >&2; exit 1; }
command -v srun >/dev/null || { echo 'srun is not available' >&2; exit 1; }

# Match the exact output path and effective config that run.py will use.
# A directory or progress.jsonl alone does not imply a finished run.
is_finished() {
  python3 - "$1" "$2" "$3" "$MAX_TOOL_CALLS" "$LUT_K" "$WEIGHT_AREA" "$WEIGHT_DELAY" <<'PY'
import json
import sys
from dataclasses import asdict
from importlib import import_module
from common.experiments import describe_experiment, experiment_path

algorithm, circuit, seed, budget, k, wa, wd = sys.argv[1:]
load_config = import_module(f"algorithms.{algorithm}.engine").load_config
config = asdict(load_config(None, seed=int(seed), max_tool_calls=int(budget),
    lut_k=int(k), optimization_mode="weighted_sum",
    weight_area=float(wa), weight_delay=float(wd)))
experiment = describe_experiment(algorithm, circuit, config)
result_path = experiment_path(experiment) / "result.json"
try:
    result = json.loads(result_path.read_text())
except (OSError, json.JSONDecodeError):
    sys.exit(1)
valid = (
    result.get("status") in ("completed", "budget_exhausted", "search_space_exhausted")
    and result.get("best") is not None
    and result.get("config") == json.loads(json.dumps(config))
    and result.get("experiment", {}).get("circuit_sha256") == experiment["circuit_sha256"]
)
sys.exit(0 if valid else 1)
PY
}

circuits=()
for dir in benchmarks/epfl/arithmetic benchmarks/epfl/random_control benchmarks/converted/iscas85; do
  [[ -d "$dir" ]] || { echo "Missing benchmark directory: $dir" >&2; exit 1; }
  while IFS= read -r -d '' file; do
    circuits+=("$file")
  done < <(find -L "$dir" -maxdepth 1 -type f -name '*.aig' -print0 | sort -z)
done
((${#circuits[@]} > 0)) || { echo 'No .aig circuits found' >&2; exit 1; }

log_root="logs/srun/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_root"
passed=0
failed=0
skipped=0

printf 'Running %s: %d experiments (%d circuits x %d seeds)\n' \
  "$algorithm" "$((${#circuits[@]} * ${#SEEDS[@]}))" "${#circuits[@]}" "${#SEEDS[@]}"

for circuit in "${circuits[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if is_finished "$algorithm" "$circuit" "$seed"; then
      ((skipped+=1))
      printf 'SKIP  %s %s seed=%s (existing valid result)\n' "$algorithm" "$circuit" "$seed"
      continue
    fi

    stem="${circuit#benchmarks/}"
    stem="${stem%.aig}"
    log="$log_root/$algorithm/$stem/seed_${seed}.log"
    mkdir -p "$(dirname "$log")"
    printf 'START %s %s seed=%s\n' "$algorithm" "$circuit" "$seed"
    if srun "${SRUN_ARGS[@]}" python3 -u run.py \
      --algorithm "$algorithm" --circuit "$circuit" --seed "$seed" \
      --max-tool-calls "$MAX_TOOL_CALLS" --lut-k "$LUT_K" \
      --optimization-mode weighted_sum \
      --weight-area "$WEIGHT_AREA" --weight-delay "$WEIGHT_DELAY" \
      >"$log" 2>&1; then
      ((passed+=1))
      printf 'PASS  %s %s seed=%s\n' "$algorithm" "$circuit" "$seed"
    else
      ((failed+=1))
      printf 'FAIL  %s %s seed=%s (see %s)\n' "$algorithm" "$circuit" "$seed" "$log" >&2
    fi
  done
done
printf 'Finished: passed=%d skipped=%d failed=%d; logs=%s\n' "$passed" "$skipped" "$failed" "$log_root"
((failed == 0))

