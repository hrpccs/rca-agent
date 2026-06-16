-- Canonical ClickHouse schema for the RCA agent (database `rca`).
--
-- Every benchmark modality is one table. A `case_id String` column is added to
-- all modality tables so the loader can scope rows per benchmark case; the
-- ClickhouseProvider always filters on it. Topology tables already carry
-- `case_id` in their natural key.
--
-- Time representations follow the raw dataset:
--   * metrics.time          — epoch microseconds (UInt64)
--   * traces.startTime/endTime/duration — epoch nanoseconds (UInt64)
--   * alerts.time_s         — epoch seconds (Int64); alerts._time_ parsed DateTime
--   * logs._time_           — DateTime parsed from the source ISO string
--   * events._time_         — DateTime parsed from the source
--
-- All statements are idempotent (CREATE TABLE IF NOT EXISTS). Re-running this
-- file never drops data.

CREATE DATABASE IF NOT EXISTS rca;

-- --------------------------------------------------------------------------- #
-- metrics
-- --------------------------------------------------------------------------- #
CREATE TABLE IF NOT EXISTS rca.metrics
(
    case_id        String,
    time           UInt64,          -- epoch microseconds
    domain         String,          -- "k8s" | "apm"
    entity_set     String,
    entity_id      String,
    entity_name    String,
    metric         String,
    value          Float64,
    metric_set_id  String,
    service        String,
    INDEX idx_entity_id entity_id TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree
ORDER BY (case_id, metric, entity_id, time);

-- --------------------------------------------------------------------------- #
-- logs
-- --------------------------------------------------------------------------- #
CREATE TABLE IF NOT EXISTS rca.logs
(
    case_id          String,
    content          String,
    `_time_`         DateTime,
    `_source_`       String,
    `_container_ip_` String,
    `_image_name_`   String,
    `_container_name_` String,
    `_pod_name_`     String,
    `_namespace_`    String,
    `_pod_uid_`      String,
    `__hostname__`   String,
    `_node_name_`    String,
    `_node_ip_`      String,
    INDEX idx_content content TYPE tokenbf_v1(8192, 3, 0) GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(`_time_`)
ORDER BY (case_id, `_time_`, `_pod_name_`);

-- --------------------------------------------------------------------------- #
-- traces
-- --------------------------------------------------------------------------- #
CREATE TABLE IF NOT EXISTS rca.traces
(
    case_id        String,
    traceId        String,
    spanId         String,
    parentSpanId   String,
    kind           String,
    spanName       String,
    startTime      UInt64,          -- epoch nanoseconds
    endTime        UInt64,          -- epoch nanoseconds
    duration       UInt64,          -- nanoseconds
    serviceName    String,
    pid            String,
    hostname       String,
    statusCode     String,
    statusMessage  String,
    resources      String,          -- JSON
    attributes     String           -- JSON
)
ENGINE = MergeTree
ORDER BY (case_id, traceId, startTime);

-- --------------------------------------------------------------------------- #
-- events (k8s events)
-- --------------------------------------------------------------------------- #
CREATE TABLE IF NOT EXISTS rca.events
(
    case_id      String,
    eventId      String,            -- JSON
    hostname     String,
    level        String,
    pod_id       String,
    pod_name     String,
    clusterId    String,
    clusterName  String,
    `_time_`     DateTime
)
ENGINE = MergeTree
ORDER BY (case_id, `_time_`, pod_name);

-- --------------------------------------------------------------------------- #
-- alerts (CNCF CloudEvents)
-- --------------------------------------------------------------------------- #
CREATE TABLE IF NOT EXISTS rca.alerts
(
    case_id          String,
    id               String,
    type             String,
    subtype          String,
    source           String,
    time             String,
    timestamp        String,
    subject          String,
    severity         String,
    status           String,
    resource         String,        -- JSON
    labels           String,        -- JSON
    annotations      String,        -- JSON
    data             String,        -- JSON
    dataschema       String,
    datacontenttype  String,
    time_s           Int64,         -- epoch seconds
    `_time_`         DateTime
)
ENGINE = MergeTree
ORDER BY (case_id, severity, time_s);

-- --------------------------------------------------------------------------- #
-- topology (graph). Natural key includes case_id.
-- --------------------------------------------------------------------------- #
CREATE TABLE IF NOT EXISTS rca.topology_entities
(
    case_id        String,
    id             String,
    type           String,
    name           String,
    first_observed String,
    last_observed  String,
    props          String           -- JSON
)
ENGINE = MergeTree
ORDER BY (case_id, type, id);

CREATE TABLE IF NOT EXISTS rca.topology_edges
(
    case_id        String,
    src            String,
    src_type       String,
    dst            String,
    dst_type       String,
    relation       String,
    first_observed String,
    last_observed  String
)
ENGINE = MergeTree
ORDER BY (case_id, src, dst);
