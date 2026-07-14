#!/usr/bin/env python3
"""Report the frozen keyword-query provenance used in the evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TextIO


DEFAULT_INPUT = Path("data/bioasq.jsonl")
PROVIDERS = ("semantic_scholar", "openalex_keyword", "google_scholar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--provider", choices=PROVIDERS)
    parser.add_argument("--question-id")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print aggregate model, source, and request-normalization counts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSONL to this path instead of stdout",
    )
    return parser.parse_args()


def keyword_records(
    path: Path,
    *,
    provider: str | None = None,
    question_id: str | None = None,
) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if question_id is not None and row["question_id"] != question_id:
                continue
            for query_provider, query in row["queries"].items():
                if provider is not None and query_provider != provider:
                    continue
                yield {
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "provider": query_provider,
                    "model": query["model"],
                    "source": query["source"],
                    "prompt": query["prompt"],
                    "prompt_sha256": query["prompt_sha256"],
                    "prompt_provenance": query["prompt_provenance"],
                    "generated_query": query["query"],
                    "request_query_used": query.get("request_query", query["query"]),
                    "query_version_id": query["query_version_id"],
                }


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    return {
        "rows": len(rows),
        "providers": dict(sorted(Counter(row["provider"] for row in rows).items())),
        "models": dict(sorted(Counter(row["model"] for row in rows).items())),
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "prompt_provenance": dict(
            sorted(Counter(row["prompt_provenance"] for row in rows).items())
        ),
        "generated_query_differs_from_request_query": sum(
            row["generated_query"] != row["request_query_used"] for row in rows
        ),
    }


def write_jsonl(records: Iterable[dict[str, Any]], handle: TextIO) -> None:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    records = keyword_records(
        args.input,
        provider=args.provider,
        question_id=args.question_id,
    )
    if args.summary:
        print(json.dumps(summarize(records), indent=2, sort_keys=True))
        return
    if args.output is None:
        write_jsonl(records, sys.stdout)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        write_jsonl(records, handle)


if __name__ == "__main__":
    main()
