-- Idempotent schema for the RCA agent application database (`rca`).
-- UTF8mb4 for full Unicode (e.g. CJK alert titles). All statements use
-- IF NOT EXISTS so this file is safe to re-run via ensure_schema().

CREATE DATABASE IF NOT EXISTS `rca`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE `rca`;

CREATE TABLE IF NOT EXISTS `cases` (
  `case_id`          VARCHAR(64)  NOT NULL,
  `task_json`        LONGTEXT     NULL,
  `topology_summary` TEXT         NULL,
  `created_at`       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`case_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `rca_runs` (
  `run_id`       VARCHAR(64)  NOT NULL,
  `case_id`      VARCHAR(64)  NULL,
  `status`       VARCHAR(32)  NULL,
  `model`        VARCHAR(64)  NULL,
  `started_at`   DATETIME     NULL,
  `finished_at`  DATETIME     NULL,
  `token_usage`  JSON         NULL,
  PRIMARY KEY (`run_id`),
  INDEX `ix_rca_runs_case_id` (`case_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `rca_reports` (
  `report_id`        VARCHAR(64)  NOT NULL,
  `run_id`           VARCHAR(64)  NULL,
  `case_id`          VARCHAR(64)  NULL,
  `alert_title`      VARCHAR(255) NULL,
  `root_cause_json`  LONGTEXT     NULL,
  `steps_json`       LONGTEXT     NULL,
  `confidence`       DOUBLE       NULL,
  `created_at`       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`report_id`),
  INDEX `ix_rca_reports_case_id` (`case_id`),
  INDEX `ix_rca_reports_run_id` (`run_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `config` (
  `kv_key`     VARCHAR(128) NOT NULL,
  `kv_value`   LONGTEXT     NULL,
  `updated_at` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                             ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`kv_key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
