CREATE TABLE `live_predictions` (
	`id` text PRIMARY KEY NOT NULL,
	`tournament_id` text NOT NULL,
	`tour` text NOT NULL,
	`tournament_name` text NOT NULL,
	`round` text NOT NULL,
	`match_id` text NOT NULL,
	`player_one` text NOT NULL,
	`player_two` text NOT NULL,
	`predicted_winner` text NOT NULL,
	`predicted_probability` real NOT NULL,
	`predicted_at` text NOT NULL,
	`starts_at` text,
	`model_version` text NOT NULL,
	`actual_winner` text,
	`correct` integer,
	`resolved_at` text
);
--> statement-breakpoint
CREATE INDEX `idx_live_predictions_tournament` ON `live_predictions` (`tournament_id`,`predicted_at`);--> statement-breakpoint
PRAGMA optimize;
