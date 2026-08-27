import unittest

from scripts.ranking_snapshot import DATABASE_RANKING_OVERRIDES, FEATURED_RANKINGS


class RankingSnapshotTests(unittest.TestCase):
    def test_featured_tours_have_complete_unique_top_sixteen(self):
        for tour in ("ATP", "WTA"):
            rows = FEATURED_RANKINGS[tour]
            self.assertEqual(len(rows), 16)
            self.assertEqual([row[2] for row in rows], list(range(1, 17)))
            self.assertEqual(len({row[0].casefold() for row in rows}), 16)
            self.assertTrue(all(row[3] > 0 for row in rows))

    def test_database_overrides_do_not_duplicate_positions(self):
        for rows in DATABASE_RANKING_OVERRIDES.values():
            positions = [row[2] for row in rows]
            self.assertEqual(len(positions), len(set(positions)))


if __name__ == "__main__":
    unittest.main()
