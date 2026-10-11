"""Research runner: NSGA-II, Greedy, DRiLLS, BOiLS and AST-GWO integrated."""

import argparse
import sys

ALGORITHMS = (
    "ast_gwo", "drills", "nsga2", "erl_ls", "boils",
    "flowtune", "cb_evo", "rl4ls", "greedy",
)


def main():
    parser = argparse.ArgumentParser(description="Logic synthesis recipe benchmark (NSGA-II, Greedy, DRiLLS, BOiLS and AST-GWO integrated).")
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--circuit", required=True, help="Input circuit path")
    parser.add_argument("--seed", type=int, help="Experiment seed (default: 0)")
    parser.add_argument("--max-tool-calls", type=int, help="ABC sequence evaluation invocation budget override; BOiLS defaults to 200")
    parser.add_argument("--config", help="Algorithm JSON-subset YAML configuration")
    for name in ("recipe-length", "population-size", "offspring-count", "generations", "lut-k", "checkpoint-interval", "episodes",
                 "n-initial", "ssk-order", "gp-training-steps", "restart-gp-training-steps", "max-training-points",
                 "candidate-pool-size", "acquisition-steps", "tr-initial-radius",
                 "tr-success-tolerance", "tr-failure-tolerance"):
        parser.add_argument("--" + name, type=int)
    for name in ("crossover-rate", "mutation-rate", "weight-area", "weight-delay", "gamma",
                 "actor-learning-rate", "critic-learning-rate", "tr-multiplier", "k-min", "k-max"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--optimization-mode", choices=("pareto", "weighted_sum"),
                        default=argparse.SUPPRESS,
                        help="Optimization mode (default: weighted_sum; config may select pareto)")
    parser.add_argument("--timeout", type=lambda value: None if value.lower() == "none" else float(value),
                        default=argparse.SUPPRESS, help="ABC seconds, or none to disable")
    parser.add_argument("--abc-binary")
    parser.add_argument("--yosys-binary", help="DRiLLS feature extraction executable")
    parser.add_argument("--device", help="DRiLLS: auto, cpu, cuda or cuda:N")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--actor-hidden", type=int, nargs=2, metavar=("H1", "H2"))
    parser.add_argument("--critic-hidden", type=int, nargs=1, metavar="H")
    parser.add_argument("--standardise", action=argparse.BooleanOptionalAction, default=None,
                        help="BOiLS: standardize Gaussian rank targets (default: enabled)")
    parser.add_argument("--objective", help="NSGA-II: lut_count,lut_levels (both minimized)")
    args = parser.parse_args()
    if args.max_tool_calls is not None and args.max_tool_calls < 0:
        parser.error("--max-tool-calls must be nonnegative")
    if args.algorithm in ("nsga2", "greedy", "drills", "boils", "ast_gwo"):
        from dataclasses import replace
        if args.algorithm == "ast_gwo":
            from algorithms.ast_gwo.engine import load_config, run, write_results
        elif args.algorithm == "boils":
            from algorithms.boils.engine import load_config, run, write_results
        elif args.algorithm == "drills":
            from algorithms.drills.engine import load_config, run, write_results
        elif args.algorithm == "greedy":
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
            if args.algorithm == "drills":
                from algorithms.drills.engine import dependencies
                try:
                    dependencies()
                except ValueError as exc:
                    print(str(exc), file=sys.stderr)
                    return 1
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
