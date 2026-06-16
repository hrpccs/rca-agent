-- ClickHouse bootstrap for the RCA agent.
--   db `rca`      : benchmark data imported by the loader (data-provider backend)
--   db `rca_otel` : observability signals exported by otel-collector (metrics/logs)
-- User `rca` owns both. The `default` user remains for ad-hoc CLI queries.

CREATE DATABASE IF NOT EXISTS rca;
CREATE DATABASE IF NOT EXISTS rca_otel;

CREATE USER IF NOT EXISTS rca IDENTIFIED WITH sha256_password BY 'rca123';
GRANT ALL PRIVILEGES ON rca.* TO rca;
GRANT ALL PRIVILEGES ON rca_otel.* TO rca;

-- Allow `rca` to create tables the otel-collector clickhouse exporter may need.
SET GLOBAL allow_experimental_object_type = 1;  -- no-op on most versions; safe ignore
