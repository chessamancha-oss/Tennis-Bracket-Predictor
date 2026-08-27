import unittest
from pathlib import Path

from scripts.ranking_snapshot import DATABASE_RANKING_OVERRIDES, FEATURED_RANKINGS

MIGRATION = (
    Path(__file__).parents[1]
    / "web"
    / "drizzle"
    / "0003_refresh_rankings_2026_08_24.sql"
)


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

    def test_forward_migration_contains_every_verified_override(self):
        migration = MIGRATION.read_text(encoding="utf-8")
        for tour, rows in DATABASE_RANKING_OVERRIDES.items():
            for _, source_name, rank, points in rows:
                search_key = " ".join(source_name.casefold().replace("-", " ").split())
                statement = (
                    f"SET rank = {rank}, ranking_points = {points} "
                    f"WHERE tour = '{tour}' AND search_key = '{search_key}'"
                )
                self.assertIn(statement, migration)


if __name__ == "__main__":
    unittest.main()
