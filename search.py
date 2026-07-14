#!/usr/bin/env python3
"""Fetch BioASQ paper-search results from the evaluated APIs.

The runner is dependency-free, resumable, and writes one JSON object per
completed question. Transient failures are retried with exponential backoff;
exhausted and terminal failures are appended to a separate sidecar.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable


PROVIDERS = (
    "elicit",
    "consensus",
    "semantic_scholar",
    "openalex_keyword",
    "openalex_semantic",
    "google_scholar",
)
S2_FIELDS = ",".join(
    [
        "paperId",
        "externalIds",
        "title",
        "year",
        "publicationDate",
        "abstract",
        "venue",
        "url",
        "citationCount",
    ]
)
OPENALEX_FIELDS = ",".join(
    [
        "id",
        "doi",
        "title",
        "display_name",
        "publication_year",
        "publication_date",
        "ids",
        "cited_by_count",
        "relevance_score",
    ]
)


class FetchError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: str | None = None,
        response: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.response = response
        self.retryable = (
            retryable
            if retryable is not None
            else status in {408, 425, 429} or (status is not None and status >= 500)
        )


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1 / requests_per_second
        self.next_request_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            request_at = max(now, self.next_request_at)
            self.next_request_at = request_at + self.interval
        delay = request_at - now
        if delay > 0:
            time.sleep(delay)

    def backoff(self, seconds: float) -> None:
        with self.lock:
            self.next_request_at = max(self.next_request_at, time.monotonic() + seconds)


@dataclass(frozen=True)
class ProviderConfig:
    api_key_env: str | None
    base_url_env: str
    default_base_url: str
    requests_per_second: float
    max_results: int


CONFIGS = {
    "elicit": ProviderConfig("ELICIT_API_KEY", "ELICIT_BASE_URL", "https://elicit.com", 2, 200),
    "consensus": ProviderConfig(
        "CONSENSUS_API_KEY", "CONSENSUS_BASE_URL", "https://api.consensus.app", 2, 20
    ),
    "semantic_scholar": ProviderConfig(
        "SEMANTIC_SCHOLAR_API_KEY",
        "SEMANTIC_SCHOLAR_BASE_URL",
        "https://api.semanticscholar.org/graph/v1",
        1,
        200,
    ),
    "openalex_keyword": ProviderConfig(
        "OPENALEX_API_KEY", "OPENALEX_BASE_URL", "https://api.openalex.org", 1, 200
    ),
    "openalex_semantic": ProviderConfig(
        "OPENALEX_API_KEY", "OPENALEX_BASE_URL", "https://api.openalex.org", 0.8, 50
    ),
    "google_scholar": ProviderConfig(
        "SERP_API__API_KEY", "SERPAPI_BASE_URL", "https://serpapi.com", 1.5, 200
    ),
}


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["question_id"] for row in read_jsonl(path)}


def terminal_failure_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    latest: dict[str, bool] = {}
    for row in read_jsonl(path):
        latest[row["question_id"]] = bool(row.get("retryable", False))
    return {question_id for question_id, retryable in latest.items() if not retryable}


def request_json(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    if params:
        encoded = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{url}{'&' if '?' in url else '?'}{encoded}"
    payload = json.dumps(body).encode() if body is not None else None
    request_headers = {"Accept": "application/json", **(headers or {})}
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=payload, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            return json.loads(text), dict(response.headers.items())
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        raise FetchError(
            f"HTTP {error.code}",
            status=error.code,
            retry_after=error.headers.get("Retry-After"),
            response=text,
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise FetchError(str(error), retryable=True) from error
    except json.JSONDecodeError as error:
        raise FetchError(f"invalid JSON response: {error}", retryable=False) from error


def retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            return max(0.0, retry_at.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


def with_retries(
    call: Callable[[], dict[str, Any]], *, attempts: int, limiter: RateLimiter
) -> dict[str, Any]:
    last_error: FetchError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except FetchError as error:
            last_error = error
            if not error.retryable or attempt == attempts:
                raise
            server_delay = retry_after_seconds(error.retry_after)
            delay = server_delay if server_delay is not None else min(60.0, 2 ** (attempt - 1))
            limiter.backoff(delay + random.uniform(0, min(1.0, delay / 4)))
    assert last_error is not None
    raise last_error


def paper(
    rank: int,
    *,
    provider_id: Any = None,
    pmid: Any = None,
    doi: Any = None,
    title: Any = None,
    year: Any = None,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "provider_id": str(provider_id) if provider_id is not None else None,
        "pmid": str(pmid) if pmid is not None else None,
        "doi": str(doi) if doi is not None else None,
        "title": str(title) if title is not None else None,
        "year": int(year) if isinstance(year, int) else None,
    }


def query_for(row: dict[str, Any], provider: str) -> str:
    if provider in {"elicit", "consensus", "openalex_semantic"}:
        return row["question"]
    query = row["queries"][provider]
    return query.get("request_query", query["query"])


def fetch_elicit(
    row: dict[str, Any], base_url: str, api_key: str, limiter: RateLimiter, timeout: float
) -> list[dict[str, Any]]:
    limiter.wait()
    payload, _ = request_json(
        "POST",
        f"{base_url}/api/v2/search/papers",
        headers={"Authorization": f"Bearer {api_key}"},
        body={
            "query": row["question"],
            "searchMode": "semantic",
            "corpus": "elicit",
            "filters": {"maxYear": row["max_year"]},
            "maxResults": 200,
        },
        timeout=timeout,
    )
    papers = payload.get("papers")
    if not isinstance(papers, list):
        raise FetchError("Elicit response has no papers list", retryable=False)
    return [
        paper(
            rank,
            provider_id=item.get("elicitId"),
            pmid=item.get("pmid"),
            doi=item.get("doi"),
            title=item.get("title"),
            year=item.get("year"),
        )
        for rank, item in enumerate(papers, start=1)
        if isinstance(item, dict)
    ]


def fetch_consensus(
    row: dict[str, Any], base_url: str, api_key: str, limiter: RateLimiter, timeout: float
) -> list[dict[str, Any]]:
    limiter.wait()
    payload, _ = request_json(
        "GET",
        f"{base_url}/v1/quick_search",
        params={"query": row["question"], "year_max": row["max_year"]},
        headers={"x-api-key": api_key},
        timeout=timeout,
    )
    results = payload.get("results")
    if not isinstance(results, list):
        raise FetchError("Consensus response has no results list", retryable=False)
    candidates = []
    for rank, item in enumerate(results, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("title"), str):
            continue
        candidates.append(
            paper(
                rank,
                provider_id=item.get("url"),
                doi=item.get("doi"),
                title=item["title"],
                year=item.get("publish_year"),
            )
        )
    return candidates


def fetch_semantic_scholar(
    row: dict[str, Any], base_url: str, api_key: str, limiter: RateLimiter, timeout: float
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    offset = 0
    while len(candidates) < 200:
        limiter.wait()
        payload, _ = request_json(
            "GET",
            f"{base_url}/paper/search",
            params={
                "query": query_for(row, "semantic_scholar"),
                "fields": S2_FIELDS,
                "limit": min(100, 200 - len(candidates)),
                "offset": offset,
                "year": f"-{row['max_year']}",
            },
            headers={"x-api-key": api_key},
            timeout=timeout,
        )
        items = payload.get("data")
        if not isinstance(items, list):
            raise FetchError("Semantic Scholar response has no data list", retryable=False)
        valid = [item for item in items if isinstance(item, dict) and item.get("title")]
        for item in valid:
            external = item.get("externalIds") or {}
            candidates.append(
                paper(
                    len(candidates) + 1,
                    provider_id=item.get("paperId"),
                    pmid=external.get("PubMed"),
                    doi=external.get("DOI"),
                    title=item.get("title"),
                    year=item.get("year"),
                )
            )
        next_offset = payload.get("next")
        if not valid or not isinstance(next_offset, int):
            break
        offset = next_offset
    return candidates


def fetch_openalex(
    row: dict[str, Any],
    provider: str,
    base_url: str,
    api_key: str,
    limiter: RateLimiter,
    timeout: float,
) -> list[dict[str, Any]]:
    semantic = provider == "openalex_semantic"
    target = 50 if semantic else 200
    cursor: str | None = None if semantic else "*"
    candidates: list[dict[str, Any]] = []
    while len(candidates) < target:
        limiter.wait()
        params: dict[str, Any] = {
            "search.semantic" if semantic else "search": query_for(row, provider),
            "filter": f"publication_year:<{row['max_year'] + 1}",
            "select": OPENALEX_FIELDS,
            "per_page": min(50 if semantic else 100, target - len(candidates)),
            "api_key": api_key,
        }
        if not semantic:
            params.update({"sort": "relevance_score:desc", "cursor": cursor})
        payload, _ = request_json("GET", f"{base_url}/works", params=params, timeout=timeout)
        items = payload.get("results")
        if not isinstance(items, list):
            raise FetchError("OpenAlex response has no results list", retryable=False)
        valid = [
            item
            for item in items
            if isinstance(item, dict) and (item.get("title") or item.get("display_name"))
        ]
        for item in valid:
            ids = item.get("ids") or {}
            raw_doi = item.get("doi") or ids.get("doi")
            raw_pmid = ids.get("pmid")
            candidates.append(
                paper(
                    len(candidates) + 1,
                    provider_id=item.get("id"),
                    pmid=(
                        raw_pmid.removeprefix("https://pubmed.ncbi.nlm.nih.gov/")
                        if isinstance(raw_pmid, str)
                        else raw_pmid
                    ),
                    doi=(
                        raw_doi.removeprefix("https://doi.org/")
                        if isinstance(raw_doi, str)
                        else raw_doi
                    ),
                    title=item.get("title") or item.get("display_name"),
                    year=item.get("publication_year"),
                )
            )
        if semantic or not valid:
            break
        meta = payload.get("meta") or {}
        cursor = meta.get("next_cursor")
        if not cursor:
            break
    return candidates


def fetch_google_scholar(
    row: dict[str, Any], base_url: str, api_key: str, limiter: RateLimiter, timeout: float
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for page_index in range(10):
        offset = page_index * 20
        limiter.wait()
        payload, _ = request_json(
            "GET",
            f"{base_url}/search.json",
            params={
                "engine": "google_scholar",
                "q": query_for(row, "google_scholar"),
                "api_key": api_key,
                "num": 20,
                "start": offset,
                "as_yhi": row["max_year"],
            },
            timeout=timeout,
        )
        if isinstance(payload.get("error"), str):
            raise FetchError(payload["error"], retryable=False)
        items = payload.get("organic_results") or []
        if not isinstance(items, list):
            raise FetchError("SerpAPI response has invalid organic_results", retryable=False)
        for raw_rank, item in enumerate(items, start=1):
            if isinstance(item, dict) and isinstance(item.get("title"), str):
                candidates.append(
                    paper(
                        offset + raw_rank,
                        provider_id=item.get("result_id"),
                        title=item["title"],
                    )
                )
        pagination = payload.get("serpapi_pagination") or {}
        if not items or not (pagination.get("next") or pagination.get("next_link")):
            break
    if not candidates:
        raise FetchError("Google Scholar returned no organic results", retryable=False)
    return candidates


def fetch_one(
    row: dict[str, Any],
    *,
    provider: str,
    base_url: str,
    api_key: str,
    limiter: RateLimiter,
    timeout: float,
    attempts: int,
) -> dict[str, Any]:
    def call() -> dict[str, Any]:
        if provider == "elicit":
            candidates = fetch_elicit(row, base_url, api_key, limiter, timeout)
        elif provider == "consensus":
            candidates = fetch_consensus(row, base_url, api_key, limiter, timeout)
        elif provider == "semantic_scholar":
            candidates = fetch_semantic_scholar(row, base_url, api_key, limiter, timeout)
        elif provider in {"openalex_keyword", "openalex_semantic"}:
            candidates = fetch_openalex(row, provider, base_url, api_key, limiter, timeout)
        else:
            candidates = fetch_google_scholar(row, base_url, api_key, limiter, timeout)
        return {
            "question_id": row["question_id"],
            "provider": provider,
            "query": query_for(row, provider),
            "max_year": row["max_year"],
            "candidates": candidates,
        }

    return with_retries(call, attempts=attempts, limiter=limiter)


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=PROVIDERS)
    parser.add_argument("--input", type=Path, default=Path("data/bioasq.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retry-terminal", action="store_true")
    parser.add_argument("--confirm-large-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    config = CONFIGS[args.provider]
    output = args.output or Path("runs") / f"{args.provider}.jsonl"
    failures = args.failures or Path("runs") / f"{args.provider}.failures.jsonl"
    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    if len(rows) > 100 and not args.confirm_large_run:
        parser.error(
            f"{len(rows)} queries requested; run a <=100-query pilot first, then pass "
            "--confirm-large-run. Google Scholar uses up to 10 paid SerpAPI calls per query."
        )
    if args.concurrency < 1 or args.attempts < 1:
        parser.error("--concurrency and --attempts must be positive")

    api_key = os.environ.get(config.api_key_env or "", "")
    if config.api_key_env and not api_key:
        parser.error(f"missing {config.api_key_env}; copy .env.example to .env")
    base_url = os.environ.get(config.base_url_env, config.default_base_url).rstrip("/")
    done = completed_ids(output)
    if not args.retry_terminal:
        done |= terminal_failure_ids(failures)
    pending = [row for row in rows if row["question_id"] not in done]
    limiter = RateLimiter(config.requests_per_second)
    write_lock = threading.Lock()
    started = time.monotonic()
    completed = 0
    successes = 0
    failed = 0

    print(
        json.dumps(
            {
                "provider": args.provider,
                "input_rows": len(rows),
                "already_done": len(rows) - len(pending),
                "pending": len(pending),
                "concurrency": args.concurrency,
                "max_results": config.max_results,
            }
        ),
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        pending_rows = iter(pending)
        future_to_row: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}

        def submit_next() -> bool:
            try:
                next_row = next(pending_rows)
            except StopIteration:
                return False
            future = executor.submit(
                fetch_one,
                next_row,
                provider=args.provider,
                base_url=base_url,
                api_key=api_key,
                limiter=limiter,
                timeout=args.timeout,
                attempts=args.attempts,
            )
            future_to_row[future] = next_row
            return True

        for _ in range(args.concurrency):
            if not submit_next():
                break
        while future_to_row:
            done_futures, _ = concurrent.futures.wait(
                future_to_row, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done_futures:
                row = future_to_row.pop(future)
                try:
                    result = future.result()
                    append_jsonl(output, result, write_lock)
                    successes += 1
                except FetchError as error:
                    append_jsonl(
                        failures,
                        {
                            "question_id": row["question_id"],
                            "provider": args.provider,
                            "query": query_for(row, args.provider),
                            "error": str(error),
                            "status": error.status,
                            "retryable": error.retryable,
                            "response": error.response,
                        },
                        write_lock,
                    )
                    failed += 1
                except Exception as error:  # preserve unexpected per-item failures
                    append_jsonl(
                        failures,
                        {
                            "question_id": row["question_id"],
                            "provider": args.provider,
                            "query": query_for(row, args.provider),
                            "error": f"{type(error).__name__}: {error}",
                            "status": None,
                            "retryable": True,
                        },
                        write_lock,
                    )
                    failed += 1
                completed += 1
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0
                eta = (len(pending) - completed) / rate if rate else None
                eta_text = f"{eta:.0f}" if eta is not None else "unknown"
                print(
                    f"completed={completed}/{len(pending)} ok={successes} failed={failed} "
                    f"eta_seconds={eta_text}",
                    flush=True,
                )
                submit_next()


if __name__ == "__main__":
    main()
