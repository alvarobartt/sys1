from __future__ import annotations

import argparse
import json
import statistics
import time

import httpx

DATASET = "LocalLLaMA/typed-decisions"
REVISION = "e135720c8fdff7896a4e4068cff624a899d41597"

def load_rows(limit: int) -> list[dict]:
    rows = []
    while len(rows) < limit:
        response = httpx.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": DATASET,
                "revision": REVISION,
                "config": "agent_trace_observability",
                "split": "train",
                "offset": len(rows),
                "length": min(100, limit - len(rows)),
            },
            timeout=60,
            follow_redirects=True,
        )
        response.raise_for_status()
        page = response.json()
        rows.extend(item["row"] for item in page["rows"])
        if not page["rows"] or len(rows) >= page["num_rows_total"]:
            break
    return rows

def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)

def main() -> None:
    parser = argparse.ArgumentParser(description="Show decision probabilities and request latency")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--api-url", help="Override --host and --port, as in benchmark.py")
    parser.add_argument("--endpoint", default="/v1/systemone")
    parser.add_argument("--api-key")
    parser.add_argument("--model", help="Optional model ID to include in requests")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = args.api_url or httpx.URL(scheme="http", host=host, port=args.port)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    rows = load_rows(args.limit)
    if not rows:
        parser.error("the dataset returned no examples")

    samples = []
    with httpx.Client(base_url=url, headers=headers, timeout=args.timeout) as client:
        for row in rows:
            state = json.loads(row["state"])
            questions = json.loads(row["questions"])
            payload = {"state": state, "questions": questions}
            if args.model:
                payload["model"] = args.model
            started = time.perf_counter()
            response = client.post(args.endpoint, json=payload)
            elapsed_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            answers = response.json()["answers"]
            columns = []
            for name, question in questions.items():
                answer = answers[name]
                if question["type"] == "noul":
                    probabilities = {"false": 1 - answer["noul"], "true": answer["noul"]}
                else:
                    probabilities = answer["probabilities"]
                columns.extend(
                    f"{name}.{label}={probabilities[label]:7.2%}"
                    for label in sorted(probabilities)
                )
            text = " ".join(state["task"].split())
            print(f"{elapsed_ms:9.2f} ms  {'  '.join(columns)}  {text}", flush=True)
            samples.append(elapsed_ms)

    print(
        f"{len(samples)} requests  mean {statistics.fmean(samples):.2f} ms  "
        f"p50 {percentile(samples, 0.50):.2f} ms  p95 {percentile(samples, 0.95):.2f} ms  "
        f"min {min(samples):.2f} ms  max {max(samples):.2f} ms"
    )

if __name__ == "__main__":
    main()
