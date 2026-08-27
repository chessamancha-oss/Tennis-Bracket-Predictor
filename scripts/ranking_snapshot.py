"""Verified official ranking snapshot used by the generated player layers."""

from __future__ import annotations

from datetime import date

RANKING_SNAPSHOT = date(2026, 8, 24)

# Display name, archive source name, official rank, and official points.
# These featured profiles are the current ATP/WTA top 16 shown by the product.
FEATURED_RANKINGS = {
    "ATP": [
        ("Jannik Sinner", "Jannik Sinner", 1, 12800),
        ("Alexander Zverev", "Alexander Zverev", 2, 7790),
        ("Carlos Alcaraz", "Carlos Alcaraz", 3, 7160),
        ("Felix Auger-Aliassime", "Felix Auger Aliassime", 4, 4640),
        ("Novak Djokovic", "Novak Djokovic", 5, 3770),
        ("Flavio Cobolli", "Flavio Cobolli", 6, 3720),
        ("Alex de Minaur", "Alex De Minaur", 7, 3650),
        ("Daniil Medvedev", "Daniil Medvedev", 8, 3580),
        ("Ben Shelton", "Ben Shelton", 9, 3480),
        ("Taylor Fritz", "Taylor Fritz", 10, 3475),
        ("Arthur Fils", "Arthur Fils", 11, 3140),
        ("Frances Tiafoe", "Frances Tiafoe", 12, 2680),
        ("Rafael Jodar", "Rafael Jodar", 13, 2671),
        ("Lorenzo Musetti", "Lorenzo Musetti", 14, 2605),
        ("Learner Tien", "Learner Tien", 15, 2565),
        ("Alexander Bublik", "Alexander Bublik", 16, 2525),
    ],
    "WTA": [
        ("Aryna Sabalenka", "Aryna Sabalenka", 1, 8575),
        ("Elena Rybakina", "Elena Rybakina", 2, 8141),
        ("Jessica Pegula", "Jessica Pegula", 3, 7265),
        ("Coco Gauff", "Coco Gauff", 4, 6704),
        ("Mirra Andreeva", "Mirra Andreeva", 5, 5443),
        ("Linda Noskova", "Linda Noskova", 6, 5028),
        ("Karolina Muchova", "Karolina Muchova", 7, 4983),
        ("Iga Swiatek", "Iga Swiatek", 8, 4809),
        ("Elina Svitolina", "Elina Svitolina", 9, 4689),
        ("Amanda Anisimova", "Amanda Anisimova", 10, 4533),
        ("Marta Kostyuk", "Marta Kostyuk", 11, 3980),
        ("Belinda Bencic", "Belinda Bencic", 12, 2995),
        ("Naomi Osaka", "Naomi Osaka", 13, 2846),
        ("Iva Jovic", "Iva Jovic", 14, 2671),
        ("Victoria Mboko", "Victoria Mboko", 15, 2531),
        ("Diana Shnaider", "Diana Shnaider", 16, 2468),
    ],
}

# Players displaced from the prior featured top 16 are retained with their
# current official position so the D1 catalogue does not preserve stale ranks.
DATABASE_RANKING_OVERRIDES = {
    "ATP": FEATURED_RANKINGS["ATP"]
    + [
        ("Jiri Lehecka", "Jiri Lehecka", 19, 2420),
        ("Casper Ruud", "Casper Ruud", 20, 2385),
        ("Andrey Rublev", "Andrey Rublev", 24, 2080),
        ("Karen Khachanov", "Karen Khachanov", 49, 1050),
    ],
    "WTA": FEATURED_RANKINGS["WTA"]
    + [("Jasmine Paolini", "Jasmine Paolini", 20, 2123)],
}
