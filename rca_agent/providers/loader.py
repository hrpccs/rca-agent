"""Parquet -> ClickHouse bulk loader (unit U2b).

Reads each benchmark case's ``.parquet`` files via pyarrow, coerces the
on-disk column names/types onto the canonical ClickHouse schema, and bulk-
inserts the rows into the matching ``rca.<table>``. Topology (entities/edges)
is loaded from ``topology.json`` via :mod:`rca_agent.cases`.

Design notes
------------
* Only programs against the foundation (``contracts``, ``config``, ``cases``).
* Idempotent at the case level: ``import_cases`` skips a case whose rows are
  already present. ``ensure_schema`` is ``CREATE TABLE IF NOT EXISTS``.
* Large tables (logs/traces can be 600k+ rows) are streamed in chunks.
* The canonical column names/types are the single source of truth for the
  ClickHouse side; the parquet side is mapped onto them with explicit renames
  and coercions. Missing parquet columns become empty/default values.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import clickhouse_connect
import pyarrow as pa
import pyarrow.parquet as pq

from ..cases import case_file, load_case, load_topology
from ..config import get_settings

if TYPE_CHECKING:
    from clickhouse_connect.driver.client import Client

logger = logging.getLogger(__name__)

# Rows per INSERT for streamed (large) tables.
CHUNK_SIZE = 50_000

# Canonical ClickHouse schema. This is the authoritative DDL for unit U2b —
# it must match the schema the downstream provider queries against. Each table
# is created IF NOT EXISTS, so it is safe to run alongside the dedicated
# schema SQL file owned by another worker unit.
_SCHEMA_DIR = Path(__file__).resolve().parent

_CANONICAL_DDL = """
CREATE TABLE IF NOT EXISTS rca.metrics
(
    case_id        String,
    time           UInt64,
    domain         String,
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

CREATE TABLE IF NOT EXISTS rca.logs
(
    case_id          String,
    content          String,
    _time_           DateTime,
    _source_         String,
    _container_ip_   String,
    _image_name_     String,
    _container_name_ String,
    _pod_name_       String,
    _namespace_      String,
    _pod_uid_        String,
    __hostname__     String,
    _node_name_      String,
    _node_ip_        String,
    INDEX idx_content content TYPE tokenbf_v1(8192, 3, 0) GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(_time_)
ORDER BY (case_id, _time_, _pod_name_);

CREATE TABLE IF NOT EXISTS rca.traces
(
    case_id       String,
    traceId       String,
    spanId        String,
    parentSpanId  String,
    kind          String,
    spanName      String,
    startTime     UInt64,
    endTime       UInt64,
    duration      UInt64,
    serviceName   String,
    pid           String,
    hostname      String,
    statusCode    String,
    statusMessage String,
    resources     String,
    attributes    String
)
ENGINE = MergeTree
ORDER BY (case_id, traceId, startTime);

CREATE TABLE IF NOT EXISTS rca.events
(
    case_id      String,
    eventId      String,
    hostname     String,
    level        String,
    pod_id       String,
    pod_name     String,
    clusterId    String,
    clusterName  String,
    _time_       DateTime
)
ENGINE = MergeTree
ORDER BY (case_id, _time_, pod_name);

CREATE TABLE IF NOT EXISTS rca.alerts
(
    case_id        String,
    id             String,
    type           String,
    subtype        String,
    source         String,
    time           String,
    timestamp      String,
    subject        String,
    severity       String,
    status         String,
    resource       String,
    labels         String,
    annotations    String,
    data           String,
    dataschema     String,
    datacontenttype String,
    time_s         Int64,
    _time_         DateTime
)
ENGINE = MergeTree
ORDER BY (case_id, severity, time_s);

CREATE TABLE IF NOT EXISTS rca.topology_entities
(
    case_id        String,
    id             String,
    type           String,
    name           String,
    first_observed String,
    last_observed  String,
    props          String
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
"""

# ---------------------------------------------------------------------------
# Per-modality column specs.
#
# Each entry maps a *canonical* (ClickHouse) column name to a coercion kind:
#   str        -> value as-is, missing -> ""
#   int        -> int(value), string ints parsed, missing/bad -> 0
#   uint64     -> same as int (CH UInt64)
#   float      -> float(value), missing/bad -> 0.0
#   datetime   -> datetime, ISO strings parsed, missing/bad -> epoch(1970)
#   json       -> dict/list serialized to JSON string
#
# `renames` maps canonical <- parquet column name when they differ.
# ---------------------------------------------------------------------------

_LOG_RENAMES = {
    "__hostname__": "__tag__:__hostname__",
    "_node_name_": "__tag__:_node_name_",
    "_node_ip_": "__tag__:_node_ip_",
}

_METRICS_COLS = {
    "time": "uint64",
    "domain": "str",
    "entity_set": "str",
    "entity_id": "str",
    "entity_name": "str",
    "metric": "str",
    "value": "float",
    "metric_set_id": "str",
    "service": "str",
}

_LOG_COLS = {
    "content": "str",
    "_time_": "datetime",
    "_source_": "str",
    "_container_ip_": "str",
    "_image_name_": "str",
    "_container_name_": "str",
    "_pod_name_": "str",
    "_namespace_": "str",
    "_pod_uid_": "str",
    "__hostname__": "str",
    "_node_name_": "str",
    "_node_ip_": "str",
}

_TRACES_COLS = {
    "traceId": "str",
    "spanId": "str",
    "parentSpanId": "str",
    "kind": "str",
    "spanName": "str",
    "startTime": "uint64",
    "endTime": "uint64",
    "duration": "uint64",
    "serviceName": "str",
    "pid": "str",
    "hostname": "str",
    "statusCode": "str",
    "statusMessage": "str",
    "resources": "str",
    "attributes": "str",
}

_EVENTS_COLS = {
    "eventId": "str",
    "hostname": "str",
    "level": "str",
    "pod_id": "str",
    "pod_name": "str",
    "clusterId": "str",
    "clusterName": "str",
    "_time_": "datetime",
}

_ALERTS_COLS = {
    "id": "str",
    "type": "str",
    "subtype": "str",
    "source": "str",
    "time": "str",
    "timestamp": "str",
    "subject": "str",
    "severity": "str",
    "status": "str",
    "resource": "str",
    "labels": "str",
    "annotations": "str",
    "data": "str",
    "dataschema": "str",
    "datacontenttype": "str",
    "time_s": "int",
    "_time_": "datetime",
}

_TOPO_ENT_COLS = {
    "id": "str",
    "type": "str",
    "name": "str",
    # first/last_observed are epoch-second ints in topology.json; the canonical
    # ClickHouse schema stores them as String (most portable across deployments),
    # so coerce to str. String-encoded ints satisfy both String and UInt64 columns.
    "first_observed": "str",
    "last_observed": "str",
    "props": "json",
}

_TOPO_EDGE_COLS = {
    "src": "str",
    "src_type": "str",
    "dst": "str",
    "dst_type": "str",
    "relation": "str",
    "first_observed": "str",
    "last_observed": "str",
}

# Parquet filename + table name per modality. Order matters only for logging.
_TABLES: dict[str, tuple[str, dict[str, str], dict[str, str] | None]] = {
    "metrics": ("metrics", _METRICS_COLS, None),
    "logs": ("logs", _LOG_COLS, _LOG_RENAMES),
    "traces": ("traces", _TRACES_COLS, None),
    "events": ("events", _EVENTS_COLS, None),
    "alerts": ("alerts", _ALERTS_COLS, None),
}

# Large tables that must be streamed in chunks.
_STREAMED = {"logs", "traces"}

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_client(database: str | None = None) -> "Client":
    """Build a clickhouse_connect client from application settings.

    Uses the HTTP port (8123 by default); clickhouse_connect is an HTTP driver,
    not the native 9000 protocol.
    """
    s = get_settings()
    return clickhouse_connect.get_client(
        host=s.clickhouse_host,
        port=s.clickhouse_port,
        username=s.clickhouse_user,
        password=s.clickhouse_password,
        database=database or s.clickhouse_database,
    )


def ensure_schema(client: "Client") -> None:
    """Create the canonical tables if they do not exist.

    Prefers ``rca_agent/providers/clickhouse_schema.sql`` (owned by another
    worker unit) when present; otherwise falls back to the inline canonical
    DDL baked into this module. Either path the canonical tables exist after.
    """
    sql_file = _SCHEMA_DIR / "clickhouse_schema.sql"
    statements: list[str]
    if sql_file.exists():
        logger.info("ensure_schema: using %s", sql_file)
        statements = [st for st in sql_file.read_text().split(";") if st.strip()]
    else:
        logger.info("ensure_schema: clickhouse_schema.sql missing, using inline DDL")
        statements = [st for st in _CANONICAL_DDL.split(";") if st.strip()]
    for st in statements:
        client.command(st)


def import_case(
    case_id: str,
    cases_dir: Path | str | None = None,
    client: "Client | None" = None,
    modalities: list[str] | None = None,
) -> dict[str, int]:
    """Import one benchmark case into ClickHouse.

    Parameters
    ----------
    case_id:
        Case directory name (e.g. ``"t001"``).
    cases_dir:
        Optional override for the cases root directory.
    client:
        Optional pre-built ClickHouse client. A fresh one is created from
        settings when omitted.
    modalities:
        Optional subset of modalities to import (e.g. ``["logs","metrics"]``).
        Defaults to all five data tables plus topology.

    Returns
    -------
    dict[str, int]
        ``{table: rows_inserted}`` for every table touched.
    """
    owns_client = client is None
    if client is None:
        client = get_client()
    try:
        ensure_schema(client)
        want = set(modalities) if modalities else set(_TABLES) | {"topology"}
        results: dict[str, int] = {}
        for modality, (table, cols, renames) in _TABLES.items():
            if modality not in want:
                continue
            results[table] = _import_parquet(
                client, case_id, cases_dir, table, modality, cols, renames
            )
        if "topology" in want:
            results["topology_entities"] = _import_topology_entities(
                client, case_id, cases_dir
            )
            results["topology_edges"] = _import_topology_edges(
                client, case_id, cases_dir
            )
        return results
    finally:
        if owns_client:
            client.close()


def import_cases(
    case_ids: list[str],
    cases_dir: Path | str | None = None,
    client: "Client | None" = None,
    modalities: list[str] | None = None,
    force: bool = False,
) -> dict[str, dict[str, int]]:
    """Import multiple cases, skipping ones already imported.

    A case is considered already imported when *any* canonical table already
    has >0 rows for that ``case_id``. Pass ``force=True`` to re-import.
    """
    owns_client = client is None
    if client is None:
        client = get_client()
    try:
        ensure_schema(client)
        out: dict[str, dict[str, int]] = {}
        for case_id in case_ids:
            if not force and _case_has_rows(client, case_id):
                logger.info("import_cases: %s already imported, skipping", case_id)
                out[case_id] = {}
                continue
            out[case_id] = import_case(case_id, cases_dir, client, modalities)
        return out
    finally:
        if owns_client:
            client.close()


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _case_has_rows(client: "Client", case_id: str) -> bool:
    """True if any canonical table already has rows for this case."""
    for table in _TABLES.values():
        try:
            n = client.query(
                f"SELECT count() FROM {table[0]} WHERE case_id = %(cid)s",
                parameters={"cid": case_id},
            ).first_item["count()"]
            if n and int(n) > 0:
                return True
        except Exception:  # noqa: BLE001 - table may not exist yet
            continue
    return False


def _import_parquet(
    client: "Client",
    case_id: str,
    cases_dir: Path | str | None,
    table: str,
    modality: str,
    cols: dict[str, str],
    renames: dict[str, str] | None,
) -> int:
    """Read a parquet file, coerce, and bulk-insert into ``table``."""
    parquet_path = case_file(case_id, f"{modality}.parquet", cases_dir)
    if not parquet_path.exists():
        logger.warning("import_case: %s.parquet missing for %s", modality, case_id)
        return 0
    table_obj = pq.read_table(parquet_path)
    return _insert_table_obj(client, table, cols, table_obj, case_id, renames)


def _insert_table_obj(
    client: "Client",
    table: str,
    cols: dict[str, str],
    table_obj: pa.Table,
    case_id: str,
    renames: dict[str, str] | None,
) -> int:
    """Coerce a pyarrow Table onto the canonical column list and insert."""
    canonical_names = list(cols.keys())
    arrays: list[pa.Array] = []
    parquet_names = {n for n in table_obj.schema.names}
    for canon in canonical_names:
        src = (renames or {}).get(canon, canon)
        if src in parquet_names:
            arr = _coerce(table_obj.column(src), cols[canon])
        else:
            # Missing column -> empty/default column of the right type.
            arr = _empty_column(cols[canon], table_obj.num_rows)
        arrays.append(arr)
    schema = pa.schema([pa.field(c, _arrow_type(cols[c])) for c in canonical_names])
    coerced = pa.table(arrays, schema=schema)
    canonical_names_with_id = ["case_id"] + canonical_names

    total = 0
    n = coerced.num_rows
    if n == 0:
        return 0
    if table in _STREAMED and n > CHUNK_SIZE:
        for start in range(0, n, CHUNK_SIZE):
            chunk = coerced.slice(start, CHUNK_SIZE)
            total += _insert_chunk(client, table, chunk, case_id, canonical_names_with_id)
    else:
        total += _insert_chunk(client, table, coerced, case_id, canonical_names_with_id)
    return total


def _insert_chunk(
    client: "Client",
    table: str,
    chunk: pa.Table,
    case_id: str,
    col_names: list[str],
) -> int:
    """Insert one chunk (pyarrow Table) with a leading case_id column.

    Data is sent column-oriented: a list of column-lists ordered to match
    ``col_names`` (so ``data[0]`` is the first column, etc.), paired with
    ``column_oriented=True``. clickhouse_connect otherwise treats the payload
    as a row sequence and indexes ``data[0]`` as the first row.
    """
    n = chunk.num_rows
    py_cols = {name: chunk.column(name).to_pylist() for name in col_names[1:]}
    data: list[list[Any]] = [[case_id] * n] + [py_cols[name] for name in col_names[1:]]
    client.insert(table, data, column_names=col_names, column_oriented=True)
    return n


# --------------------------------------------------------------------------- #
# Type coercion primitives
# --------------------------------------------------------------------------- #
def _arrow_type(kind: str) -> pa.DataType:
    if kind in {"int", "uint64"}:
        return pa.uint64()
    if kind == "float":
        return pa.float64()
    if kind == "datetime":
        return pa.timestamp("us", tz="UTC")
    if kind == "json":
        return pa.string()
    return pa.string()


def _empty_column(kind: str, n: int) -> pa.Array:
    """A fully-populated default column (never null) of the right type.

    Nulls are avoided deliberately: the canonical columns are non-Nullable, so
    every cell gets a concrete default (0 / 0.0 / "" / epoch datetime / "{}").
    """
    if kind in {"int", "uint64"}:
        return pa.array([0] * n, type=pa.uint64())
    if kind == "float":
        return pa.array([0.0] * n, type=pa.float64())
    if kind == "datetime":
        return pa.array([_EPOCH] * n, type=pa.timestamp("us", tz="UTC"))
    if kind == "json":
        return pa.array([""] * n, type=pa.string())
    return pa.array([""] * n, type=pa.string())


def _coerce(arr: pa.Array, kind: str) -> pa.Array:
    """Coerce a pyarrow array to the canonical kind, tolerating bad values."""
    if kind == "str":
        return _to_string(arr)
    if kind in {"int", "uint64"}:
        return _to_uint64(arr)
    if kind == "float":
        return _to_float(arr)
    if kind == "datetime":
        return _to_datetime(arr)
    if kind == "json":
        return _to_json(arr)
    return _to_string(arr)


def _to_string(arr: pa.Array) -> pa.Array:
    try:
        casted = arr.cast(pa.string())
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        casted = pa.array([_safe_str(v) for v in arr.to_pylist()], type=pa.string())
    # None -> "" so ClickHouse String columns never see nulls.
    out = ["" if v is None else v for v in casted.to_pylist()]
    return pa.array(out, type=pa.string())


def _to_uint64(arr: pa.Array) -> pa.Array:
    out: list[int] = []
    for v in arr.to_pylist():
        out.append(_safe_int(v))
    return pa.array(out, type=pa.uint64())


def _to_float(arr: pa.Array) -> pa.Array:
    out: list[float] = []
    for v in arr.to_pylist():
        out.append(_safe_float(v))
    return pa.array(out, type=pa.float64())


def _to_datetime(arr: pa.Array) -> pa.Array:
    out: list[datetime] = []
    for v in arr.to_pylist():
        out.append(_safe_datetime(v))
    return pa.array(out, type=pa.timestamp("us", tz="UTC"))


def _to_json(arr: pa.Array) -> pa.Array:
    out: list[str] = []
    for v in arr.to_pylist():
        if v is None:
            out.append("")
        elif isinstance(v, str):
            out.append(v if v else "")
        else:
            try:
                out.append(json.dumps(v, ensure_ascii=False, default=str))
            except (TypeError, ValueError):
                out.append(str(v))
    return pa.array(out, type=pa.string())


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def _safe_int(v: Any) -> int:
    """Coerce to a non-negative int suitable for a UInt64 column.

    Negatives clamp to 0 (UInt64 cannot represent them); garbage/None -> 0.
    """
    n: int
    if v is None:
        return 0
    if isinstance(v, bool):
        n = int(v)
    elif isinstance(v, (int, float)):
        try:
            n = int(v)
        except (OverflowError, ValueError):
            return 0
    elif isinstance(v, str):
        s = v.strip()
        if not s:
            return 0
        try:
            n = int(s)
        except ValueError:
            try:
                n = int(float(s))
            except ValueError:
                return 0
    else:
        return 0
    return n if n >= 0 else 0


def _safe_float(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except (OverflowError, ValueError):
            return 0.0
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return 0.0
        try:
            return float(s)
        except ValueError:
            return 0.0
    return 0.0


def _safe_datetime(v: Any) -> datetime:
    if v is None:
        return _EPOCH
    if isinstance(v, datetime):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)
    if isinstance(v, (int, float)):
        try:
            # Treat large ints as microseconds epoch.
            return datetime.fromtimestamp(float(v) / 1_000_000, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return _EPOCH
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return _EPOCH
        dt = _parse_iso(s)
        if dt is not None:
            return dt
        # Try numeric fallback (epoch ms / us).
        try:
            return datetime.fromtimestamp(float(s) / 1_000_000, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return _EPOCH
    return _EPOCH


def _parse_iso(s: str) -> datetime | None:
    """Parse common ISO-8601 variants seen in the dataset into tz-aware UTC."""
    s = s.strip()
    if not s:
        return None
    # Normalize trailing offset like +0800 -> +08:00 for fromisoformat.
    candidate = s
    try:
        candidate = s.replace("Z", "+00:00")
        # Fix compact offsets: ...T13:20:26+0800 -> +08:00
        if len(candidate) >= 5 and candidate[-5] in "+-" and candidate[-5:].isdigit():
            candidate = candidate[:-2] + ":" + candidate[-2:]
    except Exception:  # noqa: BLE001
        candidate = s
    # Drop sub-microsecond fractions (python only handles 6 digits).
    if "." in candidate:
        head, _, frac_zone = candidate.partition(".")
        # Split trailing non-digit (timezone) from fractional digits.
        digits = ""
        zone = ""
        for ch in frac_zone:
            if ch.isdigit() and not zone:
                digits += ch
            else:
                zone += ch
        digits = digits[:6]
        candidate = f"{head}.{digits}{zone}"
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Topology import (from topology.json via load_case/load_topology)
# --------------------------------------------------------------------------- #
def _import_topology_entities(
    client: "Client", case_id: str, cases_dir: Path | str | None
) -> int:
    topology = load_topology(case_id, cases_dir)
    cols = list(_TOPO_ENT_COLS.keys())
    col_names = ["case_id"] + cols
    data: dict[str, list[Any]] = {c: [] for c in col_names}
    for ent in topology.entities:
        data["case_id"].append(case_id)
        row = ent.model_dump()
        for c in cols:
            kind = _TOPO_ENT_COLS[c]
            data[c].append(_coerce_scalar(row.get(c), kind))
    if not data["case_id"]:
        return 0
    payload: list[list[Any]] = [data[c] for c in col_names]
    client.insert(
        "topology_entities", payload, column_names=col_names, column_oriented=True
    )
    return len(data["case_id"])


def _import_topology_edges(
    client: "Client", case_id: str, cases_dir: Path | str | None
) -> int:
    topology = load_topology(case_id, cases_dir)
    cols = list(_TOPO_EDGE_COLS.keys())
    col_names = ["case_id"] + cols
    data: dict[str, list[Any]] = {c: [] for c in col_names}
    for edge in topology.edges:
        data["case_id"].append(case_id)
        row = edge.model_dump()
        for c in cols:
            kind = _TOPO_EDGE_COLS[c]
            data[c].append(_coerce_scalar(row.get(c), kind))
    if not data["case_id"]:
        return 0
    payload = [data[c] for c in col_names]
    client.insert(
        "topology_edges", payload, column_names=col_names, column_oriented=True
    )
    return len(data["case_id"])


def _coerce_scalar(v: Any, kind: str) -> Any:
    if kind in {"int", "uint64"}:
        return _safe_int(v)
    if kind == "float":
        return _safe_float(v)
    if kind == "json":
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        try:
            return json.dumps(v, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(v)
    return _safe_str(v)


__all__ = ["get_client", "ensure_schema", "import_case", "import_cases"]
