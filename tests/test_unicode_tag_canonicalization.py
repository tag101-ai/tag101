from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tag101.tasks.sn101_reference.core.scoring.preprocessing import (  # noqa: E402
    normalize_tag,
    preprocess_responses,
)
from tag101.tasks.sn101_reference.core.scoring.tag_scorer import (  # noqa: E402
    TagScorer,
)


class UnicodeTagCanonicalizationTest(unittest.TestCase):
    def test_invisible_unicode_has_one_canonical_form(self) -> None:
        variants = (
            "\u200bbitcoin",
            "bit\u200bcoin",
            "bitcoin\u200b",
            "bit\u2060coin",
            "bitcoin\ufeff",
        )

        self.assertEqual(
            [normalize_tag(variant) for variant in variants],
            ["bitcoin"] * len(variants),
        )

    def test_canonical_duplicates_use_one_response_slot(self) -> None:
        responses = preprocess_responses(
            [["bitcoin", "bit\u200bcoin", "ethereum"]],
            n_tags_per_miner=3,
        )

        self.assertEqual(responses, [["bitcoin", "ethereum"]])

    def test_unicode_variants_share_duplicate_count(self) -> None:
        scorer = TagScorer.__new__(TagScorer)
        scorer.duplicate_penalty_enabled = True
        scorer.duplicate_penalty_k = 0.06
        scorer.duplicate_penalty_c = 80.0
        responses = [
            ["bitcoin", "ethereum", "regulation"],
            ["bit\u200bcoin", "ethereum\u2060", "regulation\ufeff"],
        ]

        result = scorer._apply_duplicate_penalty(
            miner_scores=[1.0, 1.0],
            normalized_responses=responses,
        )

        self.assertEqual(result["duplicate_counts"], [2, 2])


if __name__ == "__main__":
    unittest.main()
