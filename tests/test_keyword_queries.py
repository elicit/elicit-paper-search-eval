import json
import tempfile
import unittest
from pathlib import Path

from keyword_queries import keyword_records, summarize


class KeywordQueriesTest(unittest.TestCase):
    def test_reports_prompt_generated_query_and_request_query(self) -> None:
        row = {
            "question_id": "q1",
            "question": "What is X?",
            "queries": {
                "semantic_scholar": {
                    "model": "model-a",
                    "source": "cache",
                    "prompt": "prompt text",
                    "prompt_sha256": "abc",
                    "prompt_provenance": "stored_verbatim",
                    "query": "T-cell query",
                    "request_query": "T cell query",
                    "query_version_id": "version-1",
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            records = list(keyword_records(path))

        self.assertEqual(records[0]["prompt"], "prompt text")
        self.assertEqual(records[0]["generated_query"], "T-cell query")
        self.assertEqual(records[0]["request_query_used"], "T cell query")
        self.assertEqual(summarize(records)["generated_query_differs_from_request_query"], 1)


if __name__ == "__main__":
    unittest.main()
