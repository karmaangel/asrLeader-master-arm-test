from __future__ import annotations

import unittest
from unittest.mock import patch

import config
from model_service import FunASRService


def candidate(
    speaker: str,
    leader_id: str,
    *,
    score: float,
    speaker_score: float,
    support: int,
    margin: float,
    rank: int = 0,
) -> dict:
    return {
        "speaker": speaker,
        "leader_id": leader_id,
        "score": score,
        "speaker_score": speaker_score,
        "segment_top_avg": score,
        "support_segments": support,
        "speaker_margin": margin,
        "speaker_rank": rank,
    }


class LeaderMatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FunASRService.__new__(FunASRService)

    def test_same_leader_can_match_multiple_split_speakers(self) -> None:
        matches = self.service._select_leader_matches(
            {
                "liu": [
                    candidate(
                        "2",
                        "liu",
                        score=0.62,
                        speaker_score=0.60,
                        support=12,
                        margin=0.30,
                    ),
                    candidate(
                        "3",
                        "liu",
                        score=0.51,
                        speaker_score=0.49,
                        support=4,
                        margin=0.22,
                    ),
                    candidate(
                        "4",
                        "liu",
                        score=0.47,
                        speaker_score=0.45,
                        support=3,
                        margin=0.18,
                    ),
                ]
            }
        )

        self.assertEqual({"2", "3", "4"}, set(matches))
        self.assertTrue(all(match["leader_id"] == "liu" for match in matches.values()))

    def test_best_leader_wins_when_one_speaker_has_multiple_candidates(self) -> None:
        matches = self.service._select_leader_matches(
            {
                "liu": [
                    candidate(
                        "2",
                        "liu",
                        score=0.48,
                        speaker_score=0.46,
                        support=4,
                        margin=0.15,
                    )
                ],
                "zhu": [
                    candidate(
                        "2",
                        "zhu",
                        score=0.56,
                        speaker_score=0.52,
                        support=6,
                        margin=0.20,
                    )
                ],
            }
        )

        self.assertEqual("zhu", matches["2"]["leader_id"])

    def test_consistent_segments_accept_lower_margin(self) -> None:
        zhu = candidate(
            "4",
            "zhu",
            score=0.26335,
            speaker_score=0.25542,
            support=8,
            margin=0.04079,
        )

        with (
            patch.object(config, "LEADER_STRONG_SUPPORT_SEGMENTS", 6),
            patch.object(config, "LEADER_STRONG_SUPPORT_MIN_SCORE", 0.20),
            patch.object(config, "LEADER_STRONG_SUPPORT_MARGIN", 0.04),
        ):
            self.assertTrue(self.service._candidate_has_evidence(zhu, 0.45))

    def test_weak_consistent_segments_still_fail(self) -> None:
        weak = candidate(
            "5",
            "zhu",
            score=0.11452,
            speaker_score=0.03826,
            support=8,
            margin=0.05039,
        )

        self.assertFalse(self.service._candidate_has_evidence(weak, 0.45))

    def test_short_split_cluster_can_match_with_strong_speaker_score(self) -> None:
        short_split = candidate(
            "3",
            "liu",
            score=0.37,
            speaker_score=0.37,
            support=2,
            margin=0.25,
        )

        self.assertTrue(self.service._candidate_has_evidence(short_split, 0.45))

    def test_matched_speaker_labels_are_merged_and_renumbered(self) -> None:
        segments = [
            {"speaker": "2", "leader_id": "liu"},
            {"speaker": "5", "leader_id": None},
            {"speaker": "3", "leader_id": "liu"},
            {"speaker": "4", "leader_id": "liu"},
        ]
        matches = {
            "2": {"leader_id": "liu", "score": 0.62},
            "3": {"leader_id": "liu", "score": 0.51},
            "4": {"leader_id": "liu", "score": 0.47},
        }

        self.service._merge_matched_leader_speakers(segments, matches)

        self.assertEqual(segments[0]["speaker"], segments[2]["speaker"])
        self.assertEqual(segments[0]["speaker"], segments[3]["speaker"])
        self.assertNotEqual(segments[0]["speaker"], segments[1]["speaker"])


if __name__ == "__main__":
    unittest.main()
