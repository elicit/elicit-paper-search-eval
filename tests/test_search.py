import unittest
from unittest.mock import patch

from search import (
    FetchError,
    RateLimiter,
    fetch_consensus,
    fetch_elicit,
    fetch_google_scholar,
    fetch_openalex,
    fetch_semantic_scholar,
    query_for,
)


ROW = {
    "question_id": "golden_enriched:q1",
    "question": "What works?",
    "max_year": 2020,
    "queries": {
        "semantic_scholar": {
            "query": "generated-query",
            "request_query": "generated query",
        },
        "openalex_keyword": {"query": "generated AND query"},
        "google_scholar": {"query": '"generated query"'},
    },
}


class SearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.limiter = RateLimiter(1_000_000)

    def test_query_for_uses_exact_request_query(self) -> None:
        self.assertEqual(query_for(ROW, "semantic_scholar"), "generated query")
        self.assertEqual(query_for(ROW, "openalex_semantic"), "What works?")

    def test_all_provider_parsers(self) -> None:
        responses = [
            (
                {
                    "papers": [
                        {
                            "elicitId": "e1",
                            "pmid": "1",
                            "doi": "10/e",
                            "title": "Elicit",
                            "year": 2020,
                        }
                    ]
                },
                {},
            ),
            (
                {
                    "results": [
                        {
                            "url": "https://consensus.app/papers/details/c1/",
                            "doi": "10/c",
                            "title": "Consensus",
                            "publish_year": 2019,
                        }
                    ]
                },
                {},
            ),
            (
                {
                    "data": [
                        {
                            "paperId": "s1",
                            "externalIds": {"PubMed": "2", "DOI": "10/s"},
                            "title": "Semantic Scholar",
                            "year": 2018,
                        }
                    ]
                },
                {},
            ),
            (
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "ids": {
                                "pmid": "https://pubmed.ncbi.nlm.nih.gov/3",
                                "doi": "https://doi.org/10/o",
                            },
                            "title": "OpenAlex keyword",
                            "publication_year": 2017,
                        }
                    ],
                    "meta": {"next_cursor": None},
                },
                {},
            ),
            (
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W2",
                            "ids": {},
                            "title": "OpenAlex semantic",
                            "publication_year": 2016,
                        }
                    ]
                },
                {},
            ),
            (
                {"organic_results": [{"result_id": "g1", "title": "Google Scholar"}]},
                {},
            ),
        ]
        with patch("search.request_json", side_effect=responses) as request:
            self.assertEqual(
                fetch_elicit(ROW, "https://elicit.com", "key", self.limiter, 1)[0]["pmid"],
                "1",
            )
            self.assertEqual(
                fetch_consensus(ROW, "https://consensus", "key", self.limiter, 1)[0]["doi"],
                "10/c",
            )
            self.assertEqual(
                fetch_semantic_scholar(ROW, "https://semanticscholar", "key", self.limiter, 1)[0][
                    "provider_id"
                ],
                "s1",
            )
            self.assertEqual(
                fetch_openalex(
                    ROW,
                    "openalex_keyword",
                    "https://openalex",
                    "key",
                    self.limiter,
                    1,
                )[0]["pmid"],
                "3",
            )
            self.assertEqual(
                fetch_openalex(
                    ROW,
                    "openalex_semantic",
                    "https://openalex",
                    "key",
                    self.limiter,
                    1,
                )[0]["title"],
                "OpenAlex semantic",
            )
            self.assertEqual(
                fetch_google_scholar(ROW, "https://serpapi", "key", self.limiter, 1)[0]["title"],
                "Google Scholar",
            )
        semantic_scholar_params = request.call_args_list[2].kwargs["params"]
        openalex_keyword_params = request.call_args_list[3].kwargs["params"]
        self.assertEqual(semantic_scholar_params["query"], "generated query")
        self.assertEqual(openalex_keyword_params["per_page"], 100)

    def test_all_http_5xx_are_retryable(self) -> None:
        self.assertTrue(FetchError("server", status=599).retryable)
        self.assertFalse(FetchError("bad request", status=400).retryable)


if __name__ == "__main__":
    unittest.main()
