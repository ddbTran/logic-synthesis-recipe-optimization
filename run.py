"""Research runner: NSGA-II and Sequential Forward Greedy integrated."""

import argparse
import sys

ALGORITHMS = (
    "ast_gwo", "drills", "nsga2", "erl_ls", "boils",
    "flowtune", "cb_evo", "rl4ls", "greedy",
)


def main():
    parser = argparse.ArgumentParser(description="Logic synthesis recipe benchmark (NSGA-II and Greedy integrated).")
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--circuit", required=True, help="Input circuit path")
    parser.add_argument("--seed", type=int, help="Experiment seed (default: 0)")
    parser.add_argument("--max-tool-calls", type=int, help="ABC sequence evaluation invocation budget override; BOiLS requires an explicit budget")
    parser.add_argument("--config", help="Algorithm JSON-subset YAML configuration")
    for name in ("recipe-length", "population-size", "offspring-count", "generations", "lut-k", "checkpoint-interval"):
        parser.add_argument("--" + name, type=int)
    for name in ("crossover-rate", "mutation-rate", "weight-area", "weight-delay"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--optimization-mode", choices=("pareto", "weighted_sum"),
                        default=argparse.SUPPRESS,
                        help="Optimization mode (default: weighted_sum; config may select pareto)")
    parser.add_argument("--timeout", type=lambda value: None if value.lower() == "none" else float(value),
                        default=argparse.SUPPRESS, help="ABC seconds, or none to disable")
    parser.add_argument("--abc-binary")
    parser.add_argument("--objective", help="NSGA-II: lut_count,lut_levels (both minimized)")
    args = parser.parse_args()
    if args.max_tool_calls is not None and args.max_tool_calls < 0:
        parser.error("--max-tool-calls must be nonnegative")
    if args.algorithm in ("nsga2", "greedy"):
        from dataclasses import replace
        if args.algorithm == "greedy":
            from algorithms.greedy.engine import load_config, run, write_results
        else:
            from algorithms.nsga2.engine import load_config, run, write_results

        overrides = {key: value for key, value in vars(args).items()
                     if key not in ("algorithm", "circuit", "config", "timeout")}
        try:
            config = load_config(args.config, **overrides)
            if hasattr(args, "timeout"):
                config = replace(config, timeout=args.timeout)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
        from dataclasses import asdict
        from common.experiments import describe_experiment, prepare_run
        from common.progress import ProgressLogger

        experiment = describe_experiment(args.algorithm, args.circuit, asdict(config))
        directory = prepare_run(experiment)
        context = {"algorithm": args.algorithm, "circuit": experiment["circuit_identity"],
                   "effective_config": experiment["config"], "results_directory": str(directory)}
        with ProgressLogger(directory / "progress.jsonl", context=context) as progress:
            result = run(args.circuit, config, progress=progress)
        result["experiment"] = experiment
        write_results(result, directory=directory)
        print(f"{args.algorithm}: {result['status']}; tool_calls={result['tool_calls']}; results={directory}", flush=True)
        if result["failure"]:
            print(result["failure"]["message"], file=sys.stderr)
        return 0 if result["status"] != "failed" and result["best"] is not None else 1
    print(f"{args.algorithm}: not implemented; no ABC execution or results generated.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
