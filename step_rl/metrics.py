"""Aggregate measured outcomes. Grounding accuracy requires human labels."""
import argparse
import json
from pathlib import Path


def summarize(rows):
    if not rows:
        raise ValueError("No evaluation trajectories")
    events = [event for row in rows for event in row["events"]]
    labeled = [event for event in events if type(event.get("human_valid")) is bool]
    return {"episodes": len(rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "mean_duration_seconds": sum(row["duration_seconds"] for row in rows) / len(rows),
            "mean_steps": len(events) / len(rows),
            "rule_valid_rate": sum(event["rule_valid"] for event in events) / len(events) if events else None,
            "human_labeled_actions": len(labeled),
            "grounding_accuracy": sum((event["grounding"] > 0) == event["human_valid"] for event in labeled) / len(labeled) if labeled else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.directory).glob("trajectories-rank*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
