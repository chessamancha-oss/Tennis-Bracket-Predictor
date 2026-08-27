ALTER TABLE `live_predictions` ADD `voided_at` text;--> statement-breakpoint
ALTER TABLE `live_predictions` ADD `void_reason` text;--> statement-breakpoint
UPDATE `live_predictions`
SET `voided_at` = '2026-08-27T16:38:39.687Z',
    `void_reason` = 'Walkover',
    `actual_winner` = NULL,
    `correct` = NULL,
    `resolved_at` = NULL
WHERE `id` = 'ATP-189-2026:184769';--> statement-breakpoint
PRAGMA optimize;
