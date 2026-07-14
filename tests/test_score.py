import unittest

from score import first_matches, normalize_doi, normalize_pmid, normalize_title


class ScoreTest(unittest.TestCase):
    def test_normalizers(self) -> None:
        self.assertEqual(normalize_pmid("https://pubmed.ncbi.nlm.nih.gov/00123/"), "123")
        self.assertEqual(normalize_doi("https://doi.org/10.1000/ABC "), "10.1000/abc")
        self.assertEqual(
            normalize_title("  A title: with—punctuation! "), "a title withpunctuation"
        )

    def test_chain_falls_through_and_credits_once(self) -> None:
        gold = [
            {"pmid": "1", "doi": "10/a", "title": "First"},
            {"pmid": "2", "doi": "10/b", "title": "Second"},
        ]
        candidates = [
            {"rank": 1, "pmid": "999", "doi": "10/A", "title": "wrong"},
            {"rank": 2, "pmid": "1", "doi": None, "title": "First"},
            {"rank": 3, "pmid": None, "doi": None, "title": "Second!"},
        ]
        self.assertEqual(
            first_matches(gold, candidates),
            {"1": (1, "doi"), "2": (3, "title")},
        )

    def test_claimed_stronger_key_does_not_fall_through(self) -> None:
        gold = [
            {"pmid": "1", "doi": "10/a", "title": "First"},
            {"pmid": "2", "doi": "10/b", "title": "Second"},
        ]
        candidates = [
            {"rank": 1, "pmid": "1", "doi": None, "title": "First"},
            {"rank": 2, "pmid": "1", "doi": "10/b", "title": "Second"},
        ]
        self.assertEqual(first_matches(gold, candidates), {"1": (1, "pmid")})

    def test_duplicate_gold_pmid_fails(self) -> None:
        gold = [
            {"pmid": "1", "doi": None, "title": "First"},
            {"pmid": "01", "doi": None, "title": "Duplicate"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate gold PMID"):
            first_matches(gold, [])


if __name__ == "__main__":
    unittest.main()
