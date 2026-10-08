#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx[http2]>=0.27,<1",
#   "huggingface-hub>=1.0,<2",
#   "tokenizers>=0.22,<2",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import random
import re
import statistics
import subprocess
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

VERSION = "0.1.0"
DATASET_API = "https://datasets-server.huggingface.co/rows"


@dataclass(frozen=True)
class DatasetSpec:
    profile: str
    dataset: str
    revision: str
    config: str
    split: str
    schema: str


PROFILES = {
    "typed-decisions": DatasetSpec(
        profile="typed-decisions",
        dataset="LocalLLaMA/typed-decisions",
        revision="e135720c8fdff7896a4e4068cff624a899d41597",
        config="all",
        split="train",
        schema="typed-decisions",
    ),
    "evalsafe-questions": DatasetSpec(
        profile="evalsafe-questions",
        dataset="typesafe/evalsafe-agent-trace-observability",
        revision="8635540973910a92465fe2bc53e195375aa6e1a8",
        config="questions",
        split="test",
        schema="evalsafe-questions",
    ),
}

LAYER_SOURCE = {
    "repository": "https://github.com/NandhaKishorM/laya",
    "revision": "6d942c92081fbc139e736bbd9ac0023223c29b7f",
}

LONG_CONTEXT_QUESTION = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts, plan upgrades",
            "other": "everything else",
        },
    }
}

LONG_CONTEXT_REQUESTS = [
    "I was charged twice for invoice 4411, please refund the duplicate payment.",
    "The dashboard crashes with an error every time I open the reports page.",
    "What would an enterprise contract for 200 seats cost per year?",
    "Please refund my subscription payment from last month, it was billed by mistake.",
    "Our API returns 500 errors since this morning and the service is down.",
    "Can you send me pricing for upgrading our plan to the business tier?",
    "Me cobraron dos veces la factura de marzo, devuélvanme el cargo duplicado.",
    "La aplicación se cierra cada vez que abro la configuración.",
    "Quisiera una cotización para un contrato anual de 50 licencias.",
    "Fui cobrado duas vezes na fatura de março, quero o reembolso da cobrança duplicada.",
    "O sistema cai toda vez que tento gerar o relatório mensal.",
    "J'ai été facturé deux fois ce mois-ci, merci de rembourser le doublon.",
    "L'application plante dès que j'ouvre la page des paramètres.",
    "Ich wurde zweimal belastet, bitte erstatten Sie die doppelte Zahlung.",
    "Die Anwendung stürzt jedes Mal ab, wenn ich die Einstellungen öffne.",
    "Was kostet ein Jahresvertrag für 100 Nutzer?",
    "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।",
    "ऐप हर बार सेटिंग्स खोलते ही बंद हो जाता है।",
    "請求書が二重に請求されました。重複分を返金してください。",
    "تم خصم المبلغ مرتين من بطاقتي، أرجو استرداد المبلغ المكرر.",
]

FILLER = (
    "Thanks for the update on the quarterly planning meeting. We reviewed the roadmap "
    "slides, discussed hiring for the design team, agreed on the offsite venue, and "
    "noted that the parking garage will be closed next week. "
)

SYNTHETIC_STATE = {
    "ticket": {
        "subject": "Payout failing",
        "messages": [
            {
                "from": "customer",
                "text": (
                    "Hi, my Stripe payouts have failed for 3 days and I am losing "
                    "sales. Please help ASAP. "
                )
                * 6,
            }
        ],
    }
}

Q_NOUL = {
    "type": "noul",
    "instructions": "Does `ticket.messages[0].text` express urgency?",
}
Q_CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {
        "billing": "payments",
        "technical": "bugs and integrations",
        "sales": "pricing",
    },
}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(samples: list[float], digits: int = 3) -> dict[str, Any]:
    if not samples:
        return {"count": 0}
    return {
        "count": len(samples),
        "p50": round(percentile(samples, 0.50), digits),
        "p95": round(percentile(samples, 0.95), digits),
        "mean": round(statistics.fmean(samples), digits),
        "min": round(min(samples), digits),
        "max": round(max(samples), digits),
        "samples": [round(value, digits) for value in samples],
    }


def compact_summary(samples: list[float]) -> dict[str, Any]:
    result = summary(samples)
    result.pop("samples", None)
    return result


def resolve_spec(args: argparse.Namespace) -> DatasetSpec:
    profile = PROFILES[args.dataset_profile]
    dataset = args.dataset or profile.dataset
    requested_revision = args.dataset_revision or profile.revision
    revision = requested_revision
    if not re.fullmatch(r"[0-9a-f]{40}", requested_revision):
        revision = HfApi().dataset_info(dataset, revision=requested_revision).sha
    return DatasetSpec(
        profile=profile.profile,
        dataset=dataset,
        revision=revision,
        config=args.dataset_config or profile.config,
        split=args.dataset_split or profile.split,
        schema=args.dataset_schema or profile.schema,
    )


async def download_rows(spec: DatasetSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        while total is None or offset < total:
            params = {
                "dataset": spec.dataset,
                "revision": spec.revision,
                "config": spec.config,
                "split": spec.split,
                "offset": offset,
                "length": 100,
            }
            for attempt in range(7):
                response = await client.get(DATASET_API, params=params)
                if response.status_code != 429:
                    break
                retry_after = response.headers.get("retry-after")
                await asyncio.sleep(
                    float(retry_after) if retry_after else min(2**attempt, 30)
                )
            response.raise_for_status()
            page = response.json()
            total = page["num_rows_total"]
            wrapped = page["rows"]
            if not wrapped:
                break
            rows.extend(item["row"] for item in wrapped)
            offset += len(wrapped)
    return rows


def normalize_rows(rows: list[dict[str, Any]], schema: str) -> list[dict[str, Any]]:
    if schema == "typed-decisions":
        return [
            {
                "id": row["id"],
                "state": json.loads(row["state"]),
                "questions": json.loads(row["questions"]),
            }
            for row in rows
        ]
    if schema != "evalsafe-questions":
        raise ValueError(f"unsupported dataset schema: {schema}")

    grouped: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for row in rows:
        key = (row["case_id"], row["state_json"])
        example = grouped.setdefault(
            key,
            {
                "id": (
                    f"{row['case_id']}:"
                    f"{hashlib.sha256(row['state_json'].encode()).hexdigest()[:12]}"
                ),
                "state": json.loads(row["state_json"]),
                "questions": OrderedDict(),
            },
        )
        question_id = (
            f"{row['node_id']}.{row['question_id']}."
            f"{str(row['question_instance_id'])[:12]}"
        )
        example["questions"][question_id] = json.loads(row["question_json"])
    return list(grouped.values())


async def load_examples(spec: DatasetSpec, cache_root: Path) -> list[dict[str, Any]]:
    slug = spec.dataset.replace("/", "--")
    path = cache_root / f"{slug}-{spec.revision}-{spec.config}-{spec.split}.json"
    if path.is_file():
        document = json.loads(path.read_text())
        if document.get("dataset") != asdict(spec):
            raise RuntimeError(f"dataset cache metadata does not match {spec}: {path}")
        rows = document["rows"]
    else:
        rows = await download_rows(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dataset": asdict(spec), "rows": rows}) + "\n")
    examples = normalize_rows(rows, spec.schema)
    if not examples:
        raise RuntimeError(f"dataset produced no examples: {spec}")
    return examples


def build_requests(
    examples: list[dict[str, Any]],
    question_count: int,
    start: int,
    max_questions_per_request: int,
) -> tuple[list[dict[str, Any]], list[str], int]:
    requests: list[dict[str, Any]] = []
    ids: list[str] = []
    remaining = question_count
    cursor = start
    while remaining:
        example = examples[cursor % len(examples)]
        cursor += 1
        questions = list(example["questions"].items())
        take = min(remaining, max_questions_per_request, len(questions))
        if not take:
            continue
        requests.append(
            {"state": example["state"], "questions": dict(questions[:take])}
        )
        ids.append(example["id"])
        remaining -= take
    return requests, ids, cursor


def load_tokenizer(model: str, revision: str) -> tuple[Tokenizer, str]:
    path = hf_hub_download(
        repo_id=model,
        filename="tokenizer/tokenizer.json",
        revision=revision,
    )
    return Tokenizer.from_file(path), path


def render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "[" + ", ".join(render_json(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{json.dumps(key, ensure_ascii=False)}: {render_json(item)}"
                for key, item in value.items()
            )
            + "}"
        )
    return render_json(value)


def render_json(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return render_value(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def token_count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def target_url(api_url: str) -> str:
    parsed = urlsplit(api_url.rstrip("/"))
    if parsed.hostname not in {"0.0.0.0", "::"}:
        return parsed.geturl()
    netloc = f"127.0.0.1:{parsed.port}" if parsed.port else "127.0.0.1"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, ""))


def parse_labels(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            raise ValueError("--label must be KEY=VALUE")
        result[key] = value
    return result


def metadata(
    args: argparse.Namespace,
    benchmark: str,
    http_versions: set[str] | None = None,
    dataset: DatasetSpec | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark,
        "benchmark_tool": "benchmark.py",
        "benchmark_tool_version": VERSION,
        "sys1_git_sha": args.sys1_sha or git_sha(),
        "labels": parse_labels(args.label),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    if hasattr(args, "api_url"):
        result["target"] = target_url(args.api_url)
    if http_versions is not None:
        result["http_versions"] = sorted(http_versions)
    if dataset is not None:
        result["dataset"] = asdict(dataset)
    return result


def client_settings(args: argparse.Namespace, connections: int) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    return {
        "http2": True,
        "headers": headers,
        "limits": httpx.Limits(
            max_connections=connections, max_keepalive_connections=connections
        ),
        "timeout": args.timeout,
        "follow_redirects": True,
    }


async def wait_until_ready(client: httpx.AsyncClient, args: argparse.Namespace) -> None:
    base = target_url(args.api_url)
    deadline = time.monotonic() + args.startup_timeout
    error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = await client.get(f"{base}{args.health_endpoint}")
            response.raise_for_status()
            return
        except httpx.HTTPError as current:
            error = current
        await asyncio.sleep(0.25)
    raise TimeoutError(f"API was not ready after {args.startup_timeout:g}s: {error}")


async def send_request(
    client: httpx.AsyncClient, endpoint: str, payload: dict[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    response = await client.post(endpoint, json=payload)
    latency_ms = (time.perf_counter() - started) * 1_000
    response.raise_for_status()
    body = response.json()
    expected = len(payload["questions"])
    actual = len(body.get("answers", {}))
    if actual != expected:
        raise RuntimeError(f"API returned {actual} answers for {expected} questions")
    return {
        "latency_ms": latency_ms,
        "http_version": response.http_version,
        "input_tokens": int(body.get("usage", {}).get("input_tokens", 0)),
    }


async def send_scenario(
    client: httpx.AsyncClient, endpoint: str, payloads: list[dict[str, Any]]
) -> dict[str, Any]:
    started = time.perf_counter()
    responses = await asyncio.gather(
        *(send_request(client, endpoint, payload) for payload in payloads)
    )
    return {
        "elapsed_ms": (time.perf_counter() - started) * 1_000,
        "input_tokens": sum(item["input_tokens"] for item in responses),
        "http_versions": {item["http_version"] for item in responses},
    }


async def dataset_latency(args: argparse.Namespace) -> dict[str, Any]:
    spec = resolve_spec(args)
    examples = await load_examples(spec, args.cache_root)
    random.Random(args.seed).shuffle(examples)
    endpoint = f"{target_url(args.api_url)}{args.endpoint}"
    cursor = 0
    versions: set[str] = set()
    scenarios: dict[str, Any] = {}
    async with httpx.AsyncClient(
        **client_settings(args, max(10, max(args.questions)))
    ) as client:
        await wait_until_ready(client, args)
        for count in sorted(set(args.questions)):
            elapsed: list[float] = []
            tokens: list[float] = []
            ids: list[str] = []
            request_counts: list[int] = []
            for iteration in range(args.warmup + args.repetitions):
                payloads, current_ids, cursor = build_requests(
                    examples, count, cursor, args.max_questions_per_request
                )
                measurement = await send_scenario(client, endpoint, payloads)
                versions.update(measurement["http_versions"])
                if iteration >= args.warmup:
                    elapsed.append(measurement["elapsed_ms"])
                    tokens.append(float(measurement["input_tokens"]))
                    ids.extend(current_ids)
                    request_counts.append(len(payloads))
            scenarios[f"{count}_questions"] = {
                "questions": count,
                "requests_per_sample": sorted(set(request_counts)),
                "scenario_ms": summary(elapsed),
                "ms_per_question": summary([value / count for value in elapsed]),
                "input_tokens": summary(tokens),
                "input_tokens_per_second": summary(
                    [token / (ms / 1_000) for token, ms in zip(tokens, elapsed)]
                ),
                "unique_examples": len(set(ids)),
            }
            print(
                f"{count:>3} questions  p50 {summary(elapsed)['p50']:>8.3f} ms  "
                f"tokens p50 {summary(tokens)['p50']:>7.0f}",
                flush=True,
            )
    return {
        "meta": {
            **metadata(args, "dataset-latency", versions, spec),
            "seed": args.seed,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "dataset_examples": len(examples),
        },
        "scenarios": scenarios,
    }


async def dataset_throughput(args: argparse.Namespace) -> dict[str, Any]:
    spec = resolve_spec(args)
    examples = await load_examples(spec, args.cache_root)
    random.Random(args.seed).shuffle(examples)
    eligible = [
        example
        for example in examples
        if len(example["questions"]) >= args.questions_per_request
    ]
    if not eligible:
        raise RuntimeError("dataset has no examples with enough questions")
    payloads: list[dict[str, Any]] = []
    ids: list[str] = []
    cursor = 0
    for _ in range(args.warmup + args.requests):
        current, current_ids, cursor = build_requests(
            eligible,
            args.questions_per_request,
            cursor,
            args.questions_per_request,
        )
        payloads.append(current[0])
        ids.extend(current_ids)

    endpoint = f"{target_url(args.api_url)}{args.endpoint}"
    versions: set[str] = set()
    scenarios: dict[str, Any] = {}
    async with httpx.AsyncClient(
        **client_settings(args, max(args.concurrency))
    ) as client:
        await wait_until_ready(client, args)
        for concurrency in sorted(set(args.concurrency)):
            semaphore = asyncio.Semaphore(concurrency)

            async def limited(payload: dict[str, Any]) -> dict[str, Any]:
                async with semaphore:
                    return await send_request(client, endpoint, payload)

            if args.warmup:
                warm = await asyncio.gather(
                    *(limited(payload) for payload in payloads[: args.warmup])
                )
                versions.update(item["http_version"] for item in warm)
            measured = payloads[args.warmup :]
            started = time.perf_counter()
            responses = await asyncio.gather(
                *(limited(payload) for payload in measured)
            )
            elapsed_s = time.perf_counter() - started
            versions.update(item["http_version"] for item in responses)
            total_tokens = sum(item["input_tokens"] for item in responses)
            request_rate = len(measured) / elapsed_s
            scenarios[f"concurrency_{concurrency}"] = {
                "concurrency": concurrency,
                "requests": len(measured),
                "questions_per_request": args.questions_per_request,
                "elapsed_ms": round(elapsed_s * 1_000, 3),
                "requests_per_second": round(request_rate, 3),
                "questions_per_second": round(
                    request_rate * args.questions_per_request, 3
                ),
                "input_tokens_per_second": round(total_tokens / elapsed_s, 3),
                "total_input_tokens": total_tokens,
                "input_tokens_per_request": summary(
                    [float(item["input_tokens"]) for item in responses]
                ),
                "request_latency_ms": summary(
                    [item["latency_ms"] for item in responses]
                ),
                "unique_examples": len(set(ids[args.warmup :])),
            }
            print(
                f"concurrency {concurrency:>3}  {request_rate:>8.2f} req/s  "
                f"{total_tokens / elapsed_s:>10.0f} input tok/s",
                flush=True,
            )
    return {
        "meta": {
            **metadata(args, "dataset-throughput", versions, spec),
            "seed": args.seed,
            "warmup_requests": args.warmup,
            "requests_per_scenario": args.requests,
            "dataset_examples": len(examples),
            "eligible_examples": len(eligible),
        },
        "scenarios": scenarios,
    }


async def dataset_stats(args: argparse.Namespace) -> dict[str, Any]:
    spec = resolve_spec(args)
    examples = await load_examples(spec, args.cache_root)
    tokenizer, tokenizer_path = load_tokenizer(
        args.tokenizer_model, args.tokenizer_revision
    )
    state_tokens: list[float] = []
    question_tokens: list[float] = []
    combined_tokens: list[float] = []
    questions_per_state: list[float] = []
    request_bytes: list[float] = []
    kinds: Counter[str] = Counter()
    option_counts: list[float] = []
    for example in examples:
        state_text = (
            render_value(example["state"]).replace("[MASK]", " ").replace("<mask>", " ")
        )
        current_state_tokens = token_count(tokenizer, state_text)
        state_tokens.append(float(current_state_tokens))
        questions_per_state.append(float(len(example["questions"])))
        request_bytes.append(
            float(
                len(
                    json.dumps(
                        {"state": example["state"], "questions": example["questions"]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode()
                )
            )
        )
        for question in example["questions"].values():
            kind = str(question.get("type", "unknown"))
            kinds[kind] += 1
            current_question_tokens = token_count(tokenizer, render_value(question))
            question_tokens.append(float(current_question_tokens))
            combined_tokens.append(
                float(current_state_tokens + current_question_tokens)
            )
            criteria = question.get("criteria")
            if isinstance(criteria, (dict, list)):
                option_counts.append(float(len(criteria)))
            elif kind == "noul":
                option_counts.append(2.0)
    return {
        "meta": metadata(args, "dataset-stats", dataset=spec),
        "tokenizer": {
            "model": args.tokenizer_model,
            "revision": args.tokenizer_revision,
            "file": "tokenizer/tokenizer.json",
            "local_path": tokenizer_path,
        },
        "dataset": {
            "examples": len(examples),
            "questions": sum(len(item["questions"]) for item in examples),
            "question_kinds": dict(sorted(kinds.items())),
            "state_tokens": compact_summary(state_tokens),
            "states_over_token_threshold": {
                str(threshold): sum(value > threshold for value in state_tokens)
                for threshold in (512, 1024, 2048, 4096, 8192)
            },
            "question_tokens": compact_summary(question_tokens),
            "state_plus_question_tokens": compact_summary(combined_tokens),
            "questions_per_state": compact_summary(questions_per_state),
            "options_per_question": compact_summary(option_counts),
            "request_json_bytes": compact_summary(request_bytes),
        },
    }


async def synthetic_latency(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = f"{target_url(args.api_url)}{args.endpoint}"
    versions: set[str] = set()
    scenarios: dict[str, Any] = {}
    async with httpx.AsyncClient(**client_settings(args, 1)) as client:
        await wait_until_ready(client, args)
        for count in sorted(set(args.questions)):
            payload = {
                "state": SYNTHETIC_STATE,
                "questions": {
                    f"q{index}": Q_NOUL if index % 2 else Q_CHOICE
                    for index in range(count)
                },
            }
            elapsed: list[float] = []
            tokens: list[float] = []
            for iteration in range(args.warmup + args.repetitions):
                measurement = await send_request(client, endpoint, payload)
                versions.add(measurement["http_version"])
                if iteration >= args.warmup:
                    elapsed.append(measurement["latency_ms"])
                    tokens.append(float(measurement["input_tokens"]))
            scenarios[f"{count}_questions"] = {
                "questions": count,
                "scenario_ms": summary(elapsed),
                "ms_per_question": summary([value / count for value in elapsed]),
                "input_tokens": summary(tokens),
                "input_tokens_per_second": summary(
                    [token / (ms / 1_000) for token, ms in zip(tokens, elapsed)]
                ),
            }
            print(f"{count:>3} questions  p50 {summary(elapsed)['p50']:>8.3f} ms")
    return {
        "meta": {
            **metadata(args, "synthetic-latency", versions),
            "source": {**LAYER_SOURCE, "path": "research/scripts/bench_latency.py"},
            "warmup": args.warmup,
            "repetitions": args.repetitions,
        },
        "scenarios": scenarios,
    }


async def long_context(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer, _ = load_tokenizer(args.tokenizer_model, args.tokenizer_revision)
    filler_tokens = token_count(tokenizer, FILLER)
    endpoint = f"{target_url(args.api_url)}{args.endpoint}"
    versions: set[str] = set()
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    async with httpx.AsyncClient(**client_settings(args, 1)) as client:
        await wait_until_ready(client, args)
        for approximate_pad in args.pad_tokens:
            repetitions = round(approximate_pad / filler_tokens)
            payloads = []
            for request in LONG_CONTEXT_REQUESTS:
                state = FILLER * repetitions
                if repetitions:
                    state += "\n\nActual request: "
                state += request
                payloads.append({"state": state, "questions": LONG_CONTEXT_QUESTION})
            await send_request(client, endpoint, payloads[0])
            elapsed: list[float] = []
            tokens: list[float] = []
            for index, payload in enumerate(payloads):
                measurement = await send_request(client, endpoint, payload)
                versions.add(measurement["http_version"])
                elapsed.append(measurement["latency_ms"])
                tokens.append(float(measurement["input_tokens"]))
                cases.append(
                    {
                        "approximate_pad_tokens": approximate_pad,
                        "request_index": index,
                        "latency_ms": round(measurement["latency_ms"], 3),
                        "input_tokens": measurement["input_tokens"],
                    }
                )
            row = {
                "approximate_pad_tokens": approximate_pad,
                "server_max_model_len": args.server_max_model_len,
                "requests": len(payloads),
                "latency_ms": summary(elapsed),
                "input_tokens": summary(tokens),
                "input_tokens_per_second": summary(
                    [token / (ms / 1_000) for token, ms in zip(tokens, elapsed)]
                ),
            }
            rows.append(row)
            print(
                f"pad {approximate_pad:>5}  max {args.server_max_model_len:>5}  "
                f"p50 {row['latency_ms']['p50']:>8.3f} ms  "
                f"tokens {row['input_tokens']['p50']:>5.0f}",
                flush=True,
            )
    return {
        "meta": {
            **metadata(args, "long-context", versions),
            "source": {
                **LAYER_SOURCE,
                "path": "research/scripts/bench_long_context.py",
            },
            "tokenizer_model": args.tokenizer_model,
            "tokenizer_revision": args.tokenizer_revision,
            "filler_tokens_per_repetition": filler_tokens,
            "note": "HTTP performance adaptation; does not measure model accuracy.",
        },
        "rows": rows,
        "cases": cases,
    }


def add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sys1-sha")
    parser.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")


def add_http(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-url", default="http://127.0.0.1:3000")
    parser.add_argument("--api-key")
    parser.add_argument("--endpoint", default="/v1/systemone")
    parser.add_argument("--health-endpoint", default="/health")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--startup-timeout", type=float, default=300)


def add_dataset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-profile", choices=sorted(PROFILES), default="typed-decisions"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split")
    parser.add_argument(
        "--dataset-schema", choices=["typed-decisions", "evalsafe-questions"]
    )
    parser.add_argument("--cache-root", type=Path, default=Path(".benchmark-cache"))


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="End-to-end HTTP benchmarks for a running sys1 server."
    )
    commands = root.add_subparsers(dest="command", required=True)

    latency = commands.add_parser("latency", help="dataset-backed latency")
    add_output(latency)
    add_http(latency)
    add_dataset(latency)
    latency.add_argument("--questions", nargs="+", type=int, default=[1, 5, 10])
    latency.add_argument("--warmup", type=int, default=1)
    latency.add_argument("--repetitions", type=int, default=5)
    latency.add_argument("--seed", type=int, default=17)
    latency.add_argument("--max-questions-per-request", type=int, default=5)

    throughput = commands.add_parser("throughput", help="dataset-backed throughput")
    add_output(throughput)
    add_http(throughput)
    add_dataset(throughput)
    throughput.add_argument(
        "--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16]
    )
    throughput.add_argument("--requests", type=int, default=100)
    throughput.add_argument("--questions-per-request", type=int, default=5)
    throughput.add_argument("--warmup", type=int, default=10)
    throughput.add_argument("--seed", type=int, default=17)

    stats = commands.add_parser("dataset-stats", help="token and shape statistics")
    add_output(stats)
    add_dataset(stats)
    stats.add_argument(
        "--tokenizer-model", default="convaiinnovations/laya-typed-decisions"
    )
    stats.add_argument(
        "--tokenizer-revision",
        default="1a793eb568e6718f15941d08f85432581df534e3",
    )

    synthetic = commands.add_parser(
        "synthetic-latency", help="Laya-style 1/5/10/50 question latency"
    )
    add_output(synthetic)
    add_http(synthetic)
    synthetic.add_argument("--questions", nargs="+", type=int, default=[1, 5, 10, 50])
    synthetic.add_argument("--warmup", type=int, default=2)
    synthetic.add_argument("--repetitions", type=int, default=10)

    long = commands.add_parser(
        "long-context", help="HTTP adaptation of Laya's long-context workload"
    )
    add_output(long)
    add_http(long)
    long.add_argument(
        "--pad-tokens", nargs="+", type=int, default=[0, 1000, 2000, 4000, 7000]
    )
    long.add_argument("--server-max-model-len", type=int, required=True)
    long.add_argument(
        "--tokenizer-model", default="convaiinnovations/laya-multilingual"
    )
    long.add_argument(
        "--tokenizer-revision",
        default="e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
    )
    return root


def validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if hasattr(args, "questions") and any(value <= 0 for value in args.questions):
        parser.error("--questions values must be positive")
    if hasattr(args, "concurrency") and any(value <= 0 for value in args.concurrency):
        parser.error("--concurrency values must be positive")
    for name in ("repetitions", "requests", "questions_per_request"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "warmup") and args.warmup < 0:
        parser.error("--warmup cannot be negative")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate(args, parser)
    runners = {
        "latency": dataset_latency,
        "throughput": dataset_throughput,
        "dataset-stats": dataset_stats,
        "synthetic-latency": synthetic_latency,
        "long-context": long_context,
    }
    result = asyncio.run(runners[args.command](args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
