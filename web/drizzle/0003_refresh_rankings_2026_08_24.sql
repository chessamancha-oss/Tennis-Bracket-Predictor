-- Forward-only production refresh for the verified official 2026-08-24
-- ATP/WTA ranking snapshot. The historical catalogue otherwise retains the
-- attributed archive's 2026-06-08 ranking positions.
UPDATE players SET rank = NULL, ranking_points = NULL
WHERE tour = 'ATP' AND rank IN (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,20,24,49);--> statement-breakpoint
UPDATE players SET rank = NULL, ranking_points = NULL
WHERE tour = 'WTA' AND rank IN (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20);--> statement-breakpoint
UPDATE players SET rank = 1, ranking_points = 12800 WHERE tour = 'ATP' AND search_key = 'jannik sinner';--> statement-breakpoint
UPDATE players SET rank = 2, ranking_points = 7790 WHERE tour = 'ATP' AND search_key = 'alexander zverev';--> statement-breakpoint
UPDATE players SET rank = 3, ranking_points = 7160 WHERE tour = 'ATP' AND search_key = 'carlos alcaraz';--> statement-breakpoint
UPDATE players SET rank = 4, ranking_points = 4640 WHERE tour = 'ATP' AND search_key = 'felix auger aliassime';--> statement-breakpoint
UPDATE players SET rank = 5, ranking_points = 3770 WHERE tour = 'ATP' AND search_key = 'novak djokovic';--> statement-breakpoint
UPDATE players SET rank = 6, ranking_points = 3720 WHERE tour = 'ATP' AND search_key = 'flavio cobolli';--> statement-breakpoint
UPDATE players SET rank = 7, ranking_points = 3650 WHERE tour = 'ATP' AND search_key = 'alex de minaur';--> statement-breakpoint
UPDATE players SET rank = 8, ranking_points = 3580 WHERE tour = 'ATP' AND search_key = 'daniil medvedev';--> statement-breakpoint
UPDATE players SET rank = 9, ranking_points = 3480 WHERE tour = 'ATP' AND search_key = 'ben shelton';--> statement-breakpoint
UPDATE players SET rank = 10, ranking_points = 3475 WHERE tour = 'ATP' AND search_key = 'taylor fritz';--> statement-breakpoint
UPDATE players SET rank = 11, ranking_points = 3140 WHERE tour = 'ATP' AND search_key = 'arthur fils';--> statement-breakpoint
UPDATE players SET rank = 12, ranking_points = 2680 WHERE tour = 'ATP' AND search_key = 'frances tiafoe';--> statement-breakpoint
UPDATE players SET rank = 13, ranking_points = 2671 WHERE tour = 'ATP' AND search_key = 'rafael jodar';--> statement-breakpoint
UPDATE players SET rank = 14, ranking_points = 2605 WHERE tour = 'ATP' AND search_key = 'lorenzo musetti';--> statement-breakpoint
UPDATE players SET rank = 15, ranking_points = 2565 WHERE tour = 'ATP' AND search_key = 'learner tien';--> statement-breakpoint
UPDATE players SET rank = 16, ranking_points = 2525 WHERE tour = 'ATP' AND search_key = 'alexander bublik';--> statement-breakpoint
UPDATE players SET rank = 19, ranking_points = 2420 WHERE tour = 'ATP' AND search_key = 'jiri lehecka';--> statement-breakpoint
UPDATE players SET rank = 20, ranking_points = 2385 WHERE tour = 'ATP' AND search_key = 'casper ruud';--> statement-breakpoint
UPDATE players SET rank = 24, ranking_points = 2080 WHERE tour = 'ATP' AND search_key = 'andrey rublev';--> statement-breakpoint
UPDATE players SET rank = 49, ranking_points = 1050 WHERE tour = 'ATP' AND search_key = 'karen khachanov';--> statement-breakpoint
UPDATE players SET rank = 1, ranking_points = 8575 WHERE tour = 'WTA' AND search_key = 'aryna sabalenka';--> statement-breakpoint
UPDATE players SET rank = 2, ranking_points = 8141 WHERE tour = 'WTA' AND search_key = 'elena rybakina';--> statement-breakpoint
UPDATE players SET rank = 3, ranking_points = 7265 WHERE tour = 'WTA' AND search_key = 'jessica pegula';--> statement-breakpoint
UPDATE players SET rank = 4, ranking_points = 6704 WHERE tour = 'WTA' AND search_key = 'coco gauff';--> statement-breakpoint
UPDATE players SET rank = 5, ranking_points = 5443 WHERE tour = 'WTA' AND search_key = 'mirra andreeva';--> statement-breakpoint
UPDATE players SET rank = 6, ranking_points = 5028 WHERE tour = 'WTA' AND search_key = 'linda noskova';--> statement-breakpoint
UPDATE players SET rank = 7, ranking_points = 4983 WHERE tour = 'WTA' AND search_key = 'karolina muchova';--> statement-breakpoint
UPDATE players SET rank = 8, ranking_points = 4809 WHERE tour = 'WTA' AND search_key = 'iga swiatek';--> statement-breakpoint
UPDATE players SET rank = 9, ranking_points = 4689 WHERE tour = 'WTA' AND search_key = 'elina svitolina';--> statement-breakpoint
UPDATE players SET rank = 10, ranking_points = 4533 WHERE tour = 'WTA' AND search_key = 'amanda anisimova';--> statement-breakpoint
UPDATE players SET rank = 11, ranking_points = 3980 WHERE tour = 'WTA' AND search_key = 'marta kostyuk';--> statement-breakpoint
UPDATE players SET rank = 12, ranking_points = 2995 WHERE tour = 'WTA' AND search_key = 'belinda bencic';--> statement-breakpoint
UPDATE players SET rank = 13, ranking_points = 2846 WHERE tour = 'WTA' AND search_key = 'naomi osaka';--> statement-breakpoint
UPDATE players SET rank = 14, ranking_points = 2671 WHERE tour = 'WTA' AND search_key = 'iva jovic';--> statement-breakpoint
UPDATE players SET rank = 15, ranking_points = 2531 WHERE tour = 'WTA' AND search_key = 'victoria mboko';--> statement-breakpoint
UPDATE players SET rank = 16, ranking_points = 2468 WHERE tour = 'WTA' AND search_key = 'diana shnaider';--> statement-breakpoint
UPDATE players SET rank = 20, ranking_points = 2123 WHERE tour = 'WTA' AND search_key = 'jasmine paolini';--> statement-breakpoint
PRAGMA optimize;
