"""Flushed terminal and JSONL events for serial experiment progress."""

import json


class CheckpointScheduler:
    """Emit measured snapshots, grouping thresholds crossed by one evaluation.

    Call after updating best/archive, and once at termination with final=True.
    Elapsed time and event metadata alone do not constitute new progress.
    """

    def __init__(self, interval=200):
        if type(interval) is not int or interval <= 0:
            raise ValueError("checkpoint_interval must be a positive integer")
        self.interval = interval
        self.next_threshold = interval
        self.last_state = None

    def checkpoint(self, snapshot, *, final=False):
        calls = snapshot["tool_calls"]
        thresholds = []
        if calls is not None:
            while self.next_threshold <= calls:
                thresholds.append(self.next_threshold)
                self.next_threshold += self.interval
        state = {key: snapshot.get(key) for key in (
            "evaluations", "tool_calls", "best_fitness", "best_lut_count",
            "best_lut_levels", "best", "pareto", "candidate_attempts", "cache_hits",
            "baseline_evaluations", "baseline_qor",
        )}
        if (thresholds or final) and state != self.last_state:
            self.last_state = state
            return {**snapshot, "event": "checkpoint", "checkpoint_thresholds": thresholds}
        return None


class ProgressLogger:
    def __init__(self, path, *, context=None):
        self.stream = path.open("w")
        self.context = context or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    def __call__(self, event):
        event = {**event, **self.context}
        self.stream.write(json.dumps(event, sort_keys=True) + "\n")
        self.stream.flush()
        if event["event"] == "configuration":
            circuit = event["circuit"].removeprefix("repo/")
            print(f"{event['algorithm']}: circuit={circuit}", flush=True)
            return
        if event["event"] not in ("baseline", "generation", "step"):
            return
        is_baseline = event["event"] == "baseline"
        best = event["baseline_qor"] if is_baseline else event["best"]
        qor = f"LUTs={best['lut_count']} levels={best['lut_levels']}" if best else "LUTs=n/a levels=n/a"
        mode = event["optimization_mode"]
        if is_baseline:
            selection = "original circuit"
        elif mode == "weighted_sum":
            fitness = event["weighted_fitness"]
            selection = f"fitness={fitness:.8g}" if fitness is not None else "fitness=n/a"
        else:
            selection = f"pareto={event['pareto_size']}"
        percent = event["budget_utilization_percent"]
        utilization = f"{percent:.2f}%" if percent is not None else "unknown"
        position = (f"depth {event['depth']}/{event['target_depth']}" if "depth" in event
                    else f"generation {event['generation']}/{event['generations']}")
        print(f"{event['algorithm']} {position}: "
              f"mode={mode}; {qor}; {selection}; "
              f"ABC calls={event['tool_calls']}/{event['max_tool_calls']} ({utilization}); "
              f"cache_hits={event['cache_hits']}; elapsed={event['elapsed_seconds']:.3f}s; "
              f"status={event['status']}", flush=True)
