#!/usr/bin/env python3
"""Score paper-search JSONL against BioASQ gold recall."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, TextIO


KS = (10, 20, 50, 100, 200)
DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)
PUBMED_URL = re.compile(
    r"^https?://pubmed\.ncbi\.nlm\.nih\.gov/(?P<pmid>\d+)/?(?:[?#].*)?$",
    re.IGNORECASE,
)


def normalize_pmid(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    match = PUBMED_URL.match(text)
    if match:
        text = match.group("pmid")
    return text.lstrip("0") or None


def normalize_doi(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    for prefix in DOI_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.strip() or None


def normalize_title(value: Any) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = "".join("" if unicodedata.category(char).startswith("P") else char for char in text)
    return " ".join(text.split()) or None


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_indexes(
    gold: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, set[str]], dict[str, set[str]]]:
    pmids: dict[str, str] = {}
    dois: dict[str, set[str]] = {}
    titles: dict[str, set[str]] = {}
    for item in gold:
        gold_id = normalize_pmid(item.get("pmid"))
        if gold_id is None:
            raise ValueError("every BioASQ gold item must have a PMID")
        if gold_id in pmids:
            raise ValueError(f"duplicate gold PMID: {gold_id}")
        pmids[gold_id] = gold_id
        doi = normalize_doi(item.get("doi"))
        title = normalize_title(item.get("title"))
        if doi is not None:
            dois.setdefault(doi, set()).add(gold_id)
        if title is not None:
            titles.setdefault(title, set()).add(gold_id)
    return pmids, dois, titles


def first_unclaimed(ids: set[str], claimed: set[str]) -> str | None:
    return next((gold_id for gold_id in sorted(ids) if gold_id not in claimed), None)


def first_matches(
    gold: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> dict[str, tuple[int, str]]:
    pmids, dois, titles = build_indexes(gold)
    ranks = [int(candidate["rank"]) for candidate in candidates]
    duplicates = [rank for rank, count in Counter(ranks).items() if count > 1]
    if duplicates or any(rank <= 0 for rank in ranks):
        raise ValueError(f"candidate ranks must be unique positive integers: {duplicates}")
    claimed: set[str] = set()
    matches: dict[str, tuple[int, str]] = {}
    for candidate in sorted(candidates, key=lambda item: int(item["rank"])):
        pmid = normalize_pmid(candidate.get("pmid"))
        if pmid is not None and pmid in pmids:
            gold_id = pmids[pmid] if pmids[pmid] not in claimed else None
            tier = "pmid" if gold_id is not None else None
        else:
            doi = normalize_doi(candidate.get("doi"))
            if doi is not None and doi in dois:
                gold_id = first_unclaimed(dois[doi], claimed)
                tier = "doi" if gold_id is not None else None
            else:
                title = normalize_title(candidate.get("title"))
                if title is not None and title in titles:
                    gold_id = first_unclaimed(titles[title], claimed)
                    tier = "title" if gold_id is not None else None
                else:
                    gold_id = None
                    tier = None
        if gold_id is not None and tier is not None:
            claimed.add(gold_id)
            matches[gold_id] = (int(candidate["rank"]), tier)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/bioasq.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("data/elicit_results.jsonl.gz"))
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict-complete", action="store_true")
    args = parser.parse_args()

    def unique_rows(path: Path) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(path):
            question_id = row["question_id"]
            if question_id in rows:
                raise ValueError(f"{path} contains duplicate question id {question_id}")
            rows[question_id] = row
        return rows

    inputs = unique_rows(args.input)
    results = unique_rows(args.results)
    failures = list(read_jsonl(args.failures)) if args.failures and args.failures.exists() else []
    missing = sorted(set(inputs) - set(results))
    latest_failure = {row["question_id"]: row for row in failures}
    failed = sorted(set(missing) & set(latest_failure))
    unattempted = sorted(set(missing) - set(latest_failure))
    unknown = sorted(set(results) - set(inputs))
    if unknown:
        raise ValueError(f"results contain {len(unknown)} unknown question ids")
    if args.strict_complete and missing:
        raise ValueError(f"missing results for {len(missing)} questions")

    per_question: list[dict[str, Any]] = []
    tier_counts: Counter[str] = Counter()
    for question_id in inputs:
        if question_id not in results:
            continue
        gold = inputs[question_id]["gold"]
        matches = first_matches(gold, results[question_id].get("candidates", []))
        tier_counts.update(tier for _, tier in matches.values())
        total_gold = len(gold)
        recall = {
            str(k): sum(1 for rank, _ in matches.values() if rank <= k) / total_gold
            if total_gold
            else 0.0
            for k in KS
        }
        per_question.append(
            {
                "question_id": question_id,
                "total_gold": total_gold,
                "recall": recall,
                "matches": {
                    gold_id: {"rank": rank, "tier": tier}
                    for gold_id, (rank, tier) in matches.items()
                },
            }
        )

    total_gold = sum(row["total_gold"] for row in per_question)
    macro = {
        str(k): sum(row["recall"][str(k)] for row in per_question) / len(per_question)
        if per_question
        else 0.0
        for k in KS
    }
    micro = {
        str(k): sum(
            1 for row in per_question for match in row["matches"].values() if match["rank"] <= k
        )
        / total_gold
        if total_gold
        else 0.0
        for k in KS
    }
    report = {
        "cutoffs": list(KS),
        "num_input_questions": len(inputs),
        "num_scored_questions": len(per_question),
        "num_missing_questions": len(missing),
        "missing_question_ids": missing,
        "num_failure_records": len(failures),
        "num_failed_questions": len(failed),
        "failed_question_ids": failed,
        "num_unattempted_questions": len(unattempted),
        "unattempted_question_ids": unattempted,
        "total_gold": total_gold,
        "macro_recall": macro,
        "micro_recall": micro,
        "match_tier_counts": dict(sorted(tier_counts.items())),
        "matching": "PMID, then DOI, then exact normalized title; one credit per gold paper",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
