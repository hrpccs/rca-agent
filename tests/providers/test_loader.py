"""Tests for the parquet -> ClickHouse loader (unit U2b).

Pure-coercion tests run unconditionally. Live-ClickHouse tests are skipped
when the server is unreachable so CI without infra still passes; run locally
with the docker-compose stack up to exercise the full import path.
"""
from __future__ import annotations

from datetime import datetime, timezone

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
