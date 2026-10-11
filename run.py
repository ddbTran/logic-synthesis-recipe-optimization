"""Command-line placeholder until engines are integrated."""

import argparse
import sys

ALGORITHMS = (
    "ast_gwo", "drills", "nsga2", "erl_ls", "boils",
    "flowtune", "cb_evo", "rl4ls", "greedy",
)


def main():
    parser = argparse.ArgumentParser(description="Logic synthesis recipe benchmark (engines planned).")
    parser.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    parser.add_argument("--circuit", required=True, help="Input circuit path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for future engines")
    parser.add_argument("--max-tool-calls", type=int, help="Action budget override; BOiLS requires an explicit budget")
    args = parser.parse_args()
    if args.max_tool_calls is not None and args.max_tool_calls < 0:
        parser.error("--max-tool-calls must be nonnegative")
    print(f"{args.algorithm}: not implemented; no ABC execution or results generated.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
