"""Flushed terminal and JSONL events for serial experiment progress."""

import json


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
            print(f"{event['algorithm']} configuration: circuit={event['circuit']}; "
                  f"results={event.get('results_directory', 'not written')}; "
                  + json.dumps(event["config"], sort_keys=True), flush=True)
            return
        best = event["best"]
        qor = f"LUTs={best['lut_count']} levels={best['lut_levels']}" if best else "LUTs=n/a levels=n/a"
        mode = event["optimization_mode"]
        if mode == "weighted_sum":
            fitness = event["weighted_fitness"]
            selection = f"fitness={fitness:.8g}" if fitness is not None else "fitness=n/a"
            if event["event"] == "baseline":
                baseline = event["baseline_qor"]
                selection += f"; baseline LUTs={baseline['lut_count']} levels={baseline['lut_levels']}"
        else:
            selection = f"pareto={event['pareto_size']}"
        percent = event["budget_utilization_percent"]
        utilization = f"{percent:.2f}%" if percent is not None else "unknown"
        print(f"{event['algorithm']} {event['event']} {event['generation']}/{event['generations']}: "
              f"circuit={event['circuit']}; "
              f"mode={mode}; {qor}; {selection}; "
              f"ABC calls={event['tool_calls']}/{event['max_tool_calls']} ({utilization}); "
              f"cache_hits={event['cache_hits']}; elapsed={event['elapsed_seconds']:.3f}s; "
              f"status={event['status']}; results={event.get('results_directory', 'not written')}", flush=True)
