"""Flushed terminal and JSONL events for serial experiment progress."""

import json


class ProgressLogger:
    def __init__(self, path):
        self.stream = path.open("w")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    def __call__(self, event):
        self.stream.write(json.dumps(event, sort_keys=True) + "\n")
        self.stream.flush()
        if event["event"] == "configuration":
            print("nsga2 configuration: " + json.dumps(event["config"], sort_keys=True), flush=True)
            return
        best = event["best"]
        qor = f"LUTs={best['lut_count']} levels={best['lut_levels']}" if best else "LUTs=n/a levels=n/a"
        percent = event["budget_utilization_percent"]
        utilization = f"{percent:.2f}%" if percent is not None else "unknown"
        print(f"nsga2 {event['event']} {event['generation']}/{event['generations']}: "
              f"{qor}; pareto={event['pareto_size']}; "
              f"ABC calls={event['tool_calls']}/{event['max_tool_calls']} ({utilization}); "
              f"cache_hits={event['cache_hits']}; elapsed={event['elapsed_seconds']:.3f}s; "
              f"status={event['status']}", flush=True)
