"""Tests for the parquet -> ClickHouse loader (unit U2b).

Pure-coercion tests run unconditionally. Live-ClickHouse tests are skipped
when the server is unreachable so CI without infra still passes; run locally
with the docker-compose stack up to exercise the full import path.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rca_agent.providers import loader


# --------------------------------------------------------------------------- #
# Pure coercion unit tests (no infra)
# --------------------------------------------------------------------------- #
class TestCoercion:
    def test_safe_int_string_ints(self):
        assert loader._safe_int("1777093092662958015") == 1777093092662958015
        assert loader._safe_int("0") == 0
        assert loader._safe_int("-5") == 0  # negatives not valid for UInt64

    def test_safe_int_garbage(self):
        assert loader._safe_int("<arms_svc_id>") == 0
        assert loader._safe_int("") == 0
        assert loader._safe_int(None) == 0
        assert loader._safe_int(3.7) == 3

    def test_safe_float(self):
        assert loader._safe_float("1.5") == 1.5
        assert loader._safe_float("") == 0.0
        assert loader._safe_float(None) == 0.0
        assert loader._safe_float(2) == 2.0

    def test_safe_str_handles_none_and_structures(self):
        assert loader._safe_str(None) == ""
        assert loader._safe_str("x") == "x"
        assert loader._safe_str({"a": 1}) == '{"a": 1}'

    def test_parse_iso_long_fractions(self):
        dt = loader._parse_iso("2026-04-25T13:03:11.009372412+08:00")
        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.utcoffset().total_seconds() == 0  # normalized to UTC

    def test_parse_iso_compact_offset(self):
        dt = loader._parse_iso("2026-04-25T13:20:26+0800")
        assert dt is not None
        # +08:00 offset -> 05:20:26 UTC
        assert dt.hour == 5 and dt.minute == 20

    def test_parse_iso_z_suffix(self):
        dt = loader._parse_iso("2026-04-25T13:20:26.123Z")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_parse_iso_empty(self):
        assert loader._parse_iso("") is None
        assert loader._parse_iso("   ") is None

    def test_safe_datetime_none(self):
        assert loader._safe_datetime(None) == loader._EPOCH
        assert loader._safe_datetime("").tzinfo is not None

    def test_safe_datetime_round_trips_naive(self):
        naive = datetime(2026, 4, 25, 5, 18, 12)
        out = loader._safe_datetime(naive)
        assert out.tzinfo is not None  # gets UTC

    def test_safe_datetime_numeric_epoch_us(self):
        out = loader._safe_datetime(1777094292716735)
        assert out.year == 2026


# --------------------------------------------------------------------------- #
# Live ClickHouse tests (skipped without infra)
# --------------------------------------------------------------------------- #
def _live_client():
    try:
        client = loader.get_client(database="rca")
        client.query("SELECT 1")
        return client
    except Exception:  # noqa: BLE001
        return None


@pytest.fixture(scope="module")
def live_client():
    client = _live_client()
    if client is None:
        pytest.skip("ClickHouse rca DB unreachable")
    yield client
    client.close()


@pytest.fixture()
def clean_case(live_client):
    """Ensure no rows for the test case id remain before/after the test."""
    case_id = "t001"
    for t in [
        "metrics",
        "logs",
        "traces",
        "events",
        "alerts",
        "topology_entities",
        "topology_edges",
    ]:
        live_client.command(f"ALTER TABLE {t} DELETE WHERE case_id = %(cid)s", {"cid": case_id})
        # wait for mutations to apply by counting
        live_client.query(f"SELECT count() FROM {t} WHERE case_id = %(cid)s", {"cid": case_id})
    yield case_id


@pytest.mark.live
class TestLiveImport:
    def test_ensure_schema_creates_tables(self, live_client):
        loader.ensure_schema(live_client)
        names = {r[0] for r in live_client.query("SHOW TABLES FROM rca").result_rows}
        for t in [
            "metrics",
            "logs",
            "traces",
            "events",
            "alerts",
            "topology_entities",
            "topology_edges",
        ]:
            assert t in names, f"table {t} missing after ensure_schema"

    def test_import_case_counts(self, live_client, clean_case):
        case_id = clean_case
        result = loader.import_case(case_id, client=live_client)
        # All seven tables must be reported.
        for t in [
            "metrics",
            "logs",
            "traces",
            "events",
            "alerts",
            "topology_entities",
            "topology_edges",
        ]:
            assert t in result, f"{t} missing from import result"
            assert result[t] > 0, f"{t} imported 0 rows"
        # Rough magnitudes vs parquet (t001): metrics ~92k, logs ~600k,
        # traces ~510k, events ~449, alerts ~10, topology 237 entities/249 edges.
        assert 90_000 <= result["metrics"] <= 95_000
        assert result["logs"] == 600_000
        assert 500_000 <= result["traces"] <= 520_000
        assert 400 <= result["events"] <= 500
        assert 5 <= result["alerts"] <= 20
        assert result["topology_entities"] == 237
        assert result["topology_edges"] == 249

    def test_import_case_db_counts_match(self, live_client, clean_case):
        case_id = clean_case
        result = loader.import_case(case_id, client=live_client)
        for table, expected in result.items():
            got = live_client.query(
                f"SELECT count() FROM {table} WHERE case_id = %(cid)s",
                parameters={"cid": case_id},
            ).first_item["count()"]
            assert got == expected, f"{table}: db count {got} != returned {expected}"

    def test_import_cases_skips_existing(self, live_client, clean_case):
        case_id = clean_case
        first = loader.import_case(case_id, client=live_client)
        assert first["metrics"] > 0
        # Second call via import_cases must skip (empty dict).
        again = loader.import_cases([case_id], client=live_client)
        assert again[case_id] == {}

    def test_import_case_modalities_subset(self, live_client, clean_case):
        case_id = clean_case
        result = loader.import_case(case_id, client=live_client, modalities=["metrics"])
        assert set(result.keys()) <= {
            "metrics",
            "topology_entities",
            "topology_edges",
        }
        assert result["metrics"] > 0

    def test_logs_datetime_coerced(self, live_client, clean_case):
        case_id = clean_case
        loader.import_case(case_id, client=live_client, modalities=["logs"])
        # _time_ must be a real DateTime, not 1970, for at least some rows.
        row = live_client.query(
            "SELECT min(_time_), max(_time_) FROM logs WHERE case_id = %(cid)s",
            parameters={"cid": case_id},
        ).first_item
        assert row["min(_time_)"].year >= 2026
        assert row["max(_time_)"].year >= 2026

    def test_traces_uint_columns(self, live_client, clean_case):
        case_id = clean_case
        loader.import_case(case_id, client=live_client, modalities=["traces"])
        row = live_client.query(
            "SELECT max(startTime), max(endTime), max(duration) "
            "FROM traces WHERE case_id = %(cid)s",
            parameters={"cid": case_id},
        ).first_item
        # String-encoded ns epochs must have been parsed to ints.
        assert row["max(startTime)"] > 1_000_000_000_000_000
        assert row["max(endTime)"] > row["max(startTime)"]
        assert row["max(duration)"] > 0


# --------------------------------------------------------------------------- #
# Offline tests — FAKE ClickHouse client (no real DB)
# --------------------------------------------------------------------------- #
class FakeQueryResult:
    """Mimic clickhouse_connect's query-result `.first_item` accessor."""

    def __init__(self, first_item: dict | None = None) -> None:
        self.first_item = first_item or {}
        self.result_rows: list[tuple] = []


class FakeClickHouseClient:
    """Records command/insert/query calls. No real ClickHouse behind it.

    The shape of the recorded args matches the subset of the
    ``clickhouse_connect`` driver API that :mod:`rca_agent.providers.loader`
    actually exercises (``command``, ``insert``, ``query``, ``close``).
    """

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.inserts: list[tuple[str, list, list[str], bool]] = []
        # Per-table query first_item responses; defaults to count()==0.
        self.query_first_item: dict[str, dict] = {}
        self.closed = False
        # When set, the next insert() into this table raises it.
        self.fail_insert: dict[str, Exception] = {}

    def command(self, sql: str, parameters: dict | None = None) -> None:
        self.commands.append(sql)

    def insert(
        self,
        table: str,
        data: list,
        column_names: list[str] | None = None,
        column_oriented: bool = False,
        **kwargs,
    ) -> None:
        if table in self.fail_insert:
            raise self.fail_insert[table]
        self.inserts.append((table, data, list(column_names or []), column_oriented))

    def query(self, sql: str, parameters: dict | None = None) -> FakeQueryResult:
        # Match _case_has_rows' SELECT count() FROM <table> ...
        lowered = sql.lower()
        first = {"count()": 0}
        for table in ("metrics", "logs", "traces", "events", "alerts"):
            if f"from {table}" in lowered and "count()" in lowered:
                first = self.query_first_item.get(table, {"count()": 0})
                break
        return FakeQueryResult(first_item=first)

    def close(self) -> None:
        self.closed = True


def _write_parquet(path: Path, columns: dict[str, list]) -> None:
    """Write a small parquet file from a {col_name: values} dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(columns)
    pq.write_table(table, path)


def _make_case_dir(
    tmp_path: Path,
    case_id: str = "t_offline",
    *,
    modalities: dict[str, dict[str, list]] | None = None,
    topology: dict | None = None,
) -> Path:
    """Build a minimal on-disk benchmark case (parquet files + topology.json)."""
    case_dir = tmp_path / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for modality, cols in (modalities or {}).items():
        _write_parquet(case_dir / f"{modality}.parquet", cols)
    (case_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": case_id,
                "alert_title": "offline test",
                "alert_window": {"start": "2026-04-25T05:18:12Z", "end": "2026-04-25T05:28:12Z"},
                "prompt_text": "rca",
                "available_modalities": list((modalities or {}).keys()),
            }
        )
    )
    topo = topology or {
        "case_id": case_id,
        "window": {"start_iso": "2026-04-25T05:18:12Z", "end_iso": "2026-04-25T05:28:12Z"},
        "entities": [
            {
                "id": "pod-1",
                "type": "pod",
                "name": "checkout-pod",
                "first_observed": 1777094292,
                "last_observed": 1777094892,
                "props": {"image": "checkout:v1"},
            }
        ],
        "edges": [
            {
                "src": "pod-1",
                "src_type": "pod",
                "dst": "svc-1",
                "dst_type": "service",
                "relation": "calls",
                "first_observed": 1777094292,
                "last_observed": 1777094892,
            }
        ],
    }
    (case_dir / "topology.json").write_text(json.dumps(topo))
    return tmp_path  # cases_dir is the parent


class TestDDLEnsureSchema:
    """ensure_schema emits one CREATE TABLE IF NOT EXISTS per canonical table."""

    def test_ensure_schema_emits_all_tables(self):
        client = FakeClickHouseClient()
        loader.ensure_schema(client)
        # All seven canonical tables must appear in the issued DDL.
        joined = " ".join(client.commands)
        for table in (
            "rca.metrics",
            "rca.logs",
            "rca.traces",
            "rca.events",
            "rca.alerts",
            "rca.topology_entities",
            "rca.topology_edges",
        ):
            assert table in joined, f"{table} missing from ensure_schema DDL"

    def test_metrics_ddl_columns_and_types(self):
        client = FakeClickHouseClient()
        loader.ensure_schema(client)
        metrics_stmt = next(c for c in client.commands if "rca.metrics" in c)
        # Columns + types from the canonical schema.
        for col_decl in (
            "case_id        String",
            "time           UInt64",
            "metric         String",
            "value          Float64",
            "entity_id      String",
        ):
            assert _normalize_ws(col_decl) in _normalize_ws(metrics_stmt), (
                f"metrics DDL missing '{col_decl}'"
            )
        # bloom index on entity_id is part of the contract.
        assert "idx_entity_id" in metrics_stmt
        assert "bloom_filter" in metrics_stmt

    def test_logs_ddl_datetime_and_partition(self):
        client = FakeClickHouseClient()
        loader.ensure_schema(client)
        logs_stmt = next(c for c in client.commands if "rca.logs" in c)
        assert "_time_         DateTime" in _normalize_ws(logs_stmt) or "DateTime" in logs_stmt
        # Logs are partitioned by day for prune efficiency.
        assert "toYYYYMMDD" in logs_stmt
        assert "tokenbf_v1" in logs_stmt  # content bloom index

    def test_traces_ddl_uint64_columns(self):
        client = FakeClickHouseClient()
        loader.ensure_schema(client)
        traces_stmt = next(c for c in client.commands if "rca.traces" in c)
        for col in ("startTime     UInt64", "endTime       UInt64", "duration      UInt64"):
            assert _normalize_ws(col) in _normalize_ws(traces_stmt), (
                f"traces DDL missing '{col}'"
            )

    def test_events_ddl_columns(self):
        client = FakeClickHouseClient()
        loader.ensure_schema(client)
        events_stmt = next(c for c in client.commands if "rca.events" in c)
        for col in ("eventId", "hostname", "level", "pod_name", "clusterId", "_time_"):
            assert col in events_stmt, f"events DDL missing column '{col}'"

    def test_alerts_ddl_int64_and_datetime(self):
        client = FakeClickHouseClient()
        loader.ensure_schema(client)
        alerts_stmt = next(c for c in client.commands if "rca.alerts" in c)
        assert "time_s         Int64" in _normalize_ws(alerts_stmt) or "Int64" in alerts_stmt
        assert "DateTime" in alerts_stmt

    def test_ensure_schema_is_idempotent(self):
        # Every CREATE TABLE / CREATE DATABASE statement must be IF NOT EXISTS
        # so re-running ensure_schema never drops existing data. Statements may
        # carry leading SQL comments, so we strip comments before keyword checks.
        client = FakeClickHouseClient()
        loader.ensure_schema(client)
        assert client.commands, "ensure_schema emitted no statements"

        def strip_comments(stmt: str) -> str:
            lines = [
                ln for ln in stmt.splitlines()
                if ln.strip() and not ln.lstrip().startswith("--")
            ]
            return _normalize_ws(" ".join(lines)).upper()

        stripped = [strip_comments(c) for c in client.commands]
        # >= 7 CREATE TABLE statements, all idempotent.
        create_tables = [s for s in stripped if s.startswith("CREATE TABLE")]
        assert len(create_tables) >= 7
        for s in create_tables:
            assert "IF NOT EXISTS" in s, f"non-idempotent CREATE TABLE: {s!r}"
        # No destructive DDL as the statement verb (DROP/TRUNCATE/DELETE).
        for s in stripped:
            assert not s.startswith(("DROP", "TRUNCATE", "DELETE")), (
                f"destructive op: {s!r}"
            )


def _normalize_ws(s: str) -> str:
    """Collapse runs of whitespace so column-decl assertions survive reformatting."""
    return " ".join(s.split())


class TestRowCoercion:
    """Feed synthetic parquet -> assert coerced rows have correct types."""

    def test_metrics_coercion_types(self, tmp_path):
        cols = {
            "time": ["1777093092662958", "1777093092662959"],
            "domain": ["k8s", "apm"],
            "entity_set": ["s1", "s2"],
            "entity_id": ["d21", "d22"],
            "entity_name": ["cpu", "mem"],
            "metric": ["cpu_usage", "mem_usage"],
            "value": ["1.5", "2.5"],
            "metric_set_id": ["ms1", "ms2"],
            "service": ["svc", "svc2"],
        }
        cases_dir = _make_case_dir(tmp_path, modalities={"metrics": cols})
        client = FakeClickHouseClient()
        result = loader.import_case("t_offline", cases_dir, client=client, modalities=["metrics"])
        assert result["metrics"] == 2
        assert len(client.inserts) == 1
        table, data, col_names, column_oriented = client.inserts[0]
        assert table == "metrics"
        assert column_oriented is True
        # case_id is the first column.
        assert col_names[0] == "case_id"
        # time -> ints (parsed from str).
        time_col = dict(zip(col_names, data, strict=True))["time"]
        assert time_col == [1777093092662958, 1777093092662959]
        # value -> floats.
        value_col = dict(zip(col_names, data, strict=True))["value"]
        assert value_col == [1.5, 2.5]

    def test_logs_datetime_coerced_from_iso_string(self, tmp_path):
        cols = {
            "content": ["error one", "error two"],
            "_time_": ["2026-04-25T13:03:11.009+08:00", "2026-04-25T13:20:26+0800"],
            "_pod_name_": ["pod-a", "pod-b"],
        }
        cases_dir = _make_case_dir(tmp_path, modalities={"logs": cols})
        client = FakeClickHouseClient()
        loader.import_case("t_offline", cases_dir, client=client, modalities=["logs"])
        _, data, col_names, _ = client.inserts[0]
        time_col = dict(zip(col_names, data, strict=True))["_time_"]
        # Both ISO strings -> tz-aware UTC datetimes (not epoch, not strings).
        for v in time_col:
            assert isinstance(v, datetime)
            assert v.tzinfo is not None
            assert v.year == 2026
        # +08:00 offset normalized to UTC: 13:03 -> 05:03.
        assert time_col[0].hour == 5

    def test_missing_columns_become_defaults_not_nulls(self, tmp_path):
        # Only 'content' is present; every other canonical column must be
        # back-filled with a non-null default of the right type.
        cols = {"content": ["hello"]}
        cases_dir = _make_case_dir(tmp_path, modalities={"logs": cols})
        client = FakeClickHouseClient()
        loader.import_case("t_offline", cases_dir, client=client, modalities=["logs"])
        _, data, col_names, _ = client.inserts[0]
        row = dict(zip(col_names, [c[0] for c in data], strict=True))
        # Missing str column -> "".
        assert row["_pod_name_"] == ""
        # Missing datetime column -> epoch datetime.
        assert isinstance(row["_time_"], datetime)
        assert row["_time_"] == loader._EPOCH

    def test_none_values_replaced_with_empty_string(self, tmp_path):
        cols = {"content": [None, "real"]}
        cases_dir = _make_case_dir(tmp_path, modalities={"logs": cols})
        client = FakeClickHouseClient()
        loader.import_case("t_offline", cases_dir, client=client, modalities=["logs"])
        _, data, col_names, _ = client.inserts[0]
        content_col = dict(zip(col_names, data, strict=True))["content"]
        # Null -> "" so ClickHouse String columns never see nulls.
        assert content_col == ["", "real"]

    def test_traces_resources_attributes_json_stringified(self, tmp_path):
        cols = {
            "traceId": ["t1"],
            "spanId": ["s1"],
            "parentSpanId": [""],
            "kind": ["client"],
            "spanName": ["span"],
            "startTime": [1777093092662958000],
            "endTime": [1777093093662958000],
            "duration": [1000000000],
            "serviceName": ["svc"],
            "pid": ["1"],
            "hostname": ["h"],
            "statusCode": ["OK"],
            "statusMessage": [""],
            "resources": [{"attr": True}],
            "attributes": [{"http.method": "GET"}],
        }
        cases_dir = _make_case_dir(tmp_path, modalities={"traces": cols})
        client = FakeClickHouseClient()
        loader.import_case("t_offline", cases_dir, client=client, modalities=["traces"])
        _, data, col_names, _ = client.inserts[0]
        d = dict(zip(col_names, [c[0] for c in data], strict=True))
        # JSON columns must be serialized to JSON strings.
        assert d["resources"] == '{"attr": true}'
        assert d["attributes"] == '{"http.method": "GET"}'


class TestBatchChunking:
    """N rows with configured batch_size B -> exactly ceil(N/B) inserts, each <=B."""

    def test_small_table_single_insert(self, tmp_path):
        # logs IS streamed, but 5 rows < CHUNK_SIZE -> single insert (the
        # streamed path only chunks when n > CHUNK_SIZE).
        cols = {"content": [f"r{i}" for i in range(5)]}
        cases_dir = _make_case_dir(tmp_path, modalities={"logs": cols})
        client = FakeClickHouseClient()
        loader.import_case("t_offline", cases_dir, client=client, modalities=["logs"])
        logs_inserts = [i for i in client.inserts if i[0] == "logs"]
        assert len(logs_inserts) == 1
        assert len(logs_inserts[0][1][0]) == 5  # case_id column has 5 rows

    def test_streamed_table_chunked(self, tmp_path, monkeypatch):
        # Force a small chunk size so we don't have to materialize 50k rows.
        monkeypatch.setattr(loader, "CHUNK_SIZE", 3)
        # logs is streamed; 7 rows / chunk 3 -> ceil(7/3) = 3 inserts.
        n = 7
        cols = {"content": [f"r{i}" for i in range(n)]}
        cases_dir = _make_case_dir(tmp_path, modalities={"logs": cols})
        client = FakeClickHouseClient()
        loader.import_case("t_offline", cases_dir, client=client, modalities=["logs"])
        logs_inserts = [i for i in client.inserts if i[0] == "logs"]
        assert len(logs_inserts) == 3
        # Each insert block has <= chunk_size rows.
        for _, data, _, _ in logs_inserts:
            assert len(data[0]) <= 3
        # Total rows across chunks equals n.
        total = sum(len(data[0]) for _, data, _, _ in logs_inserts)
        assert total == n

    def test_streamed_table_exact_multiple(self, tmp_path, monkeypatch):
        # chunk 3, 6 rows -> exactly 2 inserts, no trailing empty chunk.
        monkeypatch.setattr(loader, "CHUNK_SIZE", 3)
        cols = {"content": [f"r{i}" for i in range(6)]}
        cases_dir = _make_case_dir(tmp_path, modalities={"logs": cols})
        client = FakeClickHouseClient()
        loader.import_case("t_offline", cases_dir, client=client, modalities=["logs"])
        logs_inserts = [i for i in client.inserts if i[0] == "logs"]
        assert len(logs_inserts) == 2
        for _, data, _, _ in logs_inserts:
            assert len(data[0]) == 3

    def test_non_streamed_large_table_not_chunked(self, tmp_path, monkeypatch):
        # metrics is NOT streamed; even above chunk_size it stays one insert.
        monkeypatch.setattr(loader, "CHUNK_SIZE", 2)
        n = 5
        cols = {
            "time": [str(i) for i in range(n)],
            "metric": [f"m{i}" for i in range(n)],
            "value": [float(i) for i in range(n)],
        }
        cases_dir = _make_case_dir(tmp_path, modalities={"metrics": cols})
        client = FakeClickHouseClient()
        loader.import_case("t_offline", cases_dir, client=client, modalities=["metrics"])
        metrics_inserts = [i for i in client.inserts if i[0] == "metrics"]
        assert len(metrics_inserts) == 1
        assert len(metrics_inserts[0][1][0]) == n


class TestTopologyAndEventsIngest:
    """topology + events rows/SQL are produced correctly via the fake client."""

    def test_topology_entities_inserted(self, tmp_path):
        cases_dir = _make_case_dir(tmp_path)
        client = FakeClickHouseClient()
        result = loader.import_case(
            "t_offline", cases_dir, client=client, modalities=["topology"]
        )
        ent_inserts = [i for i in client.inserts if i[0] == "topology_entities"]
        assert len(ent_inserts) == 1
        assert result["topology_entities"] == 1
        _, data, col_names, column_oriented = ent_inserts[0]
        assert column_oriented is True
        assert col_names[0] == "case_id"
        d = dict(zip(col_names, [c[0] for c in data], strict=True))
        assert d["id"] == "pod-1"
        assert d["type"] == "pod"
        assert d["name"] == "checkout-pod"
        # props -> JSON string.
        assert d["props"] == '{"image": "checkout:v1"}'
        # first/last_observed coerced to str (canonical String column).
        assert d["first_observed"] == "1777094292"
        assert d["last_observed"] == "1777094892"

    def test_topology_edges_inserted(self, tmp_path):
        cases_dir = _make_case_dir(tmp_path)
        client = FakeClickHouseClient()
        result = loader.import_case(
            "t_offline", cases_dir, client=client, modalities=["topology"]
        )
        edge_inserts = [i for i in client.inserts if i[0] == "topology_edges"]
        assert len(edge_inserts) == 1
        assert result["topology_edges"] == 1
        _, data, col_names, _ = edge_inserts[0]
        d = dict(zip(col_names, [c[0] for c in data], strict=True))
        assert d["src"] == "pod-1"
        assert d["dst"] == "svc-1"
        assert d["relation"] == "calls"

    def test_events_ingest_path(self, tmp_path):
        cols = {
            "eventId": ["e1", "e2"],
            "hostname": ["h1", "h2"],
            "level": ["Warning", "Normal"],
            "pod_id": ["p1", "p2"],
            "pod_name": ["pod-1", "pod-2"],
            "clusterId": ["c1", "c1"],
            "clusterName": ["cl", "cl"],
            "_time_": ["2026-04-25T05:18:12Z", "2026-04-25T05:19:00Z"],
        }
        cases_dir = _make_case_dir(tmp_path, modalities={"events": cols})
        client = FakeClickHouseClient()
        result = loader.import_case("t_offline", cases_dir, client=client, modalities=["events"])
        assert result["events"] == 2
        ev_inserts = [i for i in client.inserts if i[0] == "events"]
        assert len(ev_inserts) == 1
        _, data, col_names, _ = ev_inserts[0]
        time_col = dict(zip(col_names, data, strict=True))["_time_"]
        for v in time_col:
            assert isinstance(v, datetime)
            assert v.year == 2026

    def test_empty_topology_returns_zero(self, tmp_path):
        cases_dir = _make_case_dir(
            tmp_path, topology={"case_id": "t_offline", "entities": [], "edges": []}
        )
        client = FakeClickHouseClient()
        result = loader.import_case(
            "t_offline", cases_dir, client=client, modalities=["topology"]
        )
        assert result["topology_entities"] == 0
        assert result["topology_edges"] == 0
        # No inserts issued for empty topology.
        assert [i for i in client.inserts if i[0] in ("topology_entities", "topology_edges")] == []


class TestHardenedImport:
    """A failing table is logged + skipped; the rest of the case still imports."""

    def test_one_table_failure_does_not_abort_others(self, tmp_path, caplog):
        from clickhouse_connect.driver.exceptions import OperationalError

        cols_m = {
            "time": ["1"],
            "metric": ["m"],
            "value": [1.0],
        }
        cols_l = {"content": ["x"]}
        cases_dir = _make_case_dir(
            tmp_path, modalities={"metrics": cols_m, "logs": cols_l}
        )
        client = FakeClickHouseClient()
        client.fail_insert["metrics"] = OperationalError("boom: metrics down")
        with caplog.at_level("ERROR", logger="rca_agent.providers.loader"):
            result = loader.import_case(
                "t_offline",
                cases_dir,
                client=client,
                modalities=["metrics", "logs"],
            )
        # metrics failed -> recorded as 0, but logs still imported.
        assert result["metrics"] == 0
        assert result["logs"] == 1
        # logs insert did happen.
        assert any(i[0] == "logs" for i in client.inserts)
        # Structured log names the failing table.
        assert any(
            "metrics" in rec.message and "t_offline" in rec.message and rec.levelno >= 40
            for rec in caplog.records
        ), [r.message for r in caplog.records]

    def test_operational_error_on_topology_skips_but_continues(self, tmp_path, caplog):
        from clickhouse_connect.driver.exceptions import OperationalError

        cases_dir = _make_case_dir(tmp_path)
        client = FakeClickHouseClient()
        client.fail_insert["topology_entities"] = OperationalError("connection reset")
        with caplog.at_level("ERROR", logger="rca_agent.providers.loader"):
            result = loader.import_case(
                "t_offline", cases_dir, client=client, modalities=["topology"]
            )
        # entities failed on a transient network error, edges still imported.
        assert result["topology_entities"] == 0
        assert result["topology_edges"] == 1
        assert any(
            "topology_entities" in rec.message for rec in caplog.records
        ), [r.message for r in caplog.records]

    def test_programming_error_propagates(self, tmp_path):
        # A bad-column / type-mismatch surfaces from ClickHouse as
        # ProgrammingError, which is a DatabaseError subclass but NOT
        # OperationalError — so it must propagate loud, not be swallowed.
        from clickhouse_connect.driver.exceptions import DatabaseError, ProgrammingError

        assert issubclass(ProgrammingError, DatabaseError)  # sanity: hierarchy
        cases_dir = _make_case_dir(tmp_path)
        client = FakeClickHouseClient()
        client.fail_insert["topology_entities"] = ProgrammingError(
            "Unknown column idd in table topology_entities"
        )
        with pytest.raises(ProgrammingError):
            loader.import_case(
                "t_offline", cases_dir, client=client, modalities=["topology"]
            )

    def test_non_ch_error_propagates(self, tmp_path):
        # A non-CH error (ValueError) must NOT be swallowed — only the narrow
        # retriable CH exception set is caught.
        cols = {"content": ["x"]}
        cases_dir = _make_case_dir(tmp_path, modalities={"logs": cols})
        client = FakeClickHouseClient()
        client.fail_insert["logs"] = ValueError("real bug")
        with pytest.raises(ValueError, match="real bug"):
            loader.import_case("t_offline", cases_dir, client=client, modalities=["logs"])


class TestImportCasesSkip:
    """import_cases skips an already-imported case via the fake query path."""

    def test_skips_when_rows_exist(self, tmp_path):
        cols = {"content": ["x"]}
        cases_dir = _make_case_dir(tmp_path, modalities={"logs": cols})
        client = FakeClickHouseClient()
        # Pretend metrics already has rows for this case.
        client.query_first_item["metrics"] = {"count()": 5}
        out = loader.import_cases(["t_offline"], cases_dir, client=client)
        assert out["t_offline"] == {}
        # ensure_schema still ran (CREATE TABLEs issued) but no inserts.
        assert client.commands
        assert client.inserts == []

    def test_force_reimports_even_with_existing_rows(self, tmp_path):
        cols = {"content": ["x"]}
        cases_dir = _make_case_dir(tmp_path, modalities={"logs": cols})
        client = FakeClickHouseClient()
        client.query_first_item["metrics"] = {"count()": 5}
        out = loader.import_cases(["t_offline"], cases_dir, client=client, force=True)
        assert out["t_offline"]["logs"] == 1
        assert any(i[0] == "logs" for i in client.inserts)


class TestClientLifecycle:
    """When no client is injected, a fresh one is built and closed."""

    def test_owned_client_is_closed(self, monkeypatch):
        fake = FakeClickHouseClient()
        captured: dict = {}

        def fake_get_client(database=None):
            captured["called"] = True
            return fake

        monkeypatch.setattr(loader, "get_client", fake_get_client)
        # Use a nonexistent cases dir + metrics-only modality: the missing
        # parquet returns 0 cleanly, so we only exercise schema + lifecycle.
        loader.import_case(
            "never",
            cases_dir="/nonexistent/cases",
            client=None,
            modalities=["metrics"],
        )
        assert captured["called"] is True
        assert fake.closed is True
        assert fake.commands  # ensure_schema ran

    def test_injected_client_is_not_closed(self, tmp_path):
        client = FakeClickHouseClient()
        loader.import_case(
            "never",
            cases_dir="/nonexistent/cases",
            client=client,
            modalities=["metrics"],
        )
        assert client.closed is False
        assert client.commands  # ensure_schema still ran
