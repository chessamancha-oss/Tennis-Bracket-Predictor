CREATE TABLE `players` (
	`id` text PRIMARY KEY NOT NULL,
	`tour` text NOT NULL,
	`name` text NOT NULL,
	`search_key` text NOT NULL,
	`country` text NOT NULL,
	`hand` text NOT NULL,
	`birth_year` integer,
	`career_start` integer NOT NULL,
	`career_end` integer NOT NULL,
	`rank` integer,
	`ranking_points` integer,
	`rating` real NOT NULL,
	`rating_sigma` real NOT NULL,
	`matches` integer NOT NULL,
	`wins` integer NOT NULL,
	`form_rate` real NOT NULL,
	`serve_points_won` real NOT NULL,
	`return_points_won` real NOT NULL,
	`hold_rate` real NOT NULL,
	`ace_rate` real,
	`double_fault_rate` real,
	`serve_sample` integer NOT NULL,
	`return_sample` integer NOT NULL,
	`hard_rating` real NOT NULL,
	`hard_matches` integer NOT NULL,
	`clay_rating` real NOT NULL,
	`clay_matches` integer NOT NULL,
	`grass_rating` real NOT NULL,
	`grass_matches` integer NOT NULL,
	`major_titles` integer NOT NULL,
	`last_match_date` text
);
--> statement-breakpoint
CREATE INDEX `idx_players_search_key` ON `players` (`search_key`);--> statement-breakpoint
CREATE INDEX `idx_players_tour_era` ON `players` (`tour`,`career_start`,`career_end`);--> statement-breakpoint
CREATE INDEX `idx_players_tour_rating` ON `players` (`tour`,`rating`);