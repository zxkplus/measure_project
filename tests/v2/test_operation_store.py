"""Incremental durable state, legacy migration and offline rollback acceptance."""

import json
import sqlite3

import pytest

from measure_api.v2.errors import ServiceError
from measure_api.v2.operation_store import OperationStore, export_journal


def record(identity, state="succeeded", value=1):
    return {"operation_id": identity, "operation": "image-inspections", "state": state,
            "trace_id": "trace-" + identity, "result": {"value": value}}


def test_legacy_import_and_offline_export_preserve_records_and_retry_metadata(tmp_path):
    saved = {"records": {"first": record("first")},
             "metadata": {"first": {"key": ["caller", "image-inspections", "retry"],
                                    "fingerprint": "hash", "terminal_at": 123}},
             "expired": {"old": 124}, "expired_traces": {"old-trace": 125}}
    original = json.dumps(saved)
    (tmp_path / "journal.json").write_text(original)
    store = OperationStore(tmp_path)
    assert store.load() == saved
    store.save({"second": (record("second"), {"terminal_at": 126})})
    store.close()
    assert (tmp_path / "journal.json").read_text() == original
    exported = tmp_path / "rollback.json"
    export_journal(tmp_path, exported)
    restored = json.loads(exported.read_text())
    assert restored["records"]["first"] == saved["records"]["first"]
    assert restored["metadata"]["first"] == saved["metadata"]["first"]
    assert restored["expired"] == saved["expired"]
    assert restored["expired_traces"] == saved["expired_traces"]
    assert restored["records"]["second"] == record("second")


def test_update_does_not_reencode_unrelated_historical_results(tmp_path):
    store = OperationStore(tmp_path)
    store.load()
    store.save({"history": (record("history", value="x" * 1_000_000), {}),
                "current": (record("current", "queued"), {})})
    before = store.connection.total_changes
    metrics = store.save({"current": (record("current", value=2), {})})
    assert store.connection.total_changes - before == 1
    assert metrics["encoded_bytes"] < 1000
    assert store.load()["records"]["history"]["result"]["value"] == "x" * 1_000_000
    store.close()


def test_expiration_deletes_record_and_preserves_explicit_markers(tmp_path):
    store = OperationStore(tmp_path)
    store.load()
    store.save({"first": (record("first"), {})})
    store.save({}, removed={"first"}, markers={
        ("operation", "first"): 100, ("trace", "trace-first"): 100})
    restored = store.load()
    assert not restored["records"] and not restored["metadata"]
    assert restored["expired"] == {"first": 100}
    store.save({}, markers={("operation", "first"): None})
    assert store.load()["expired"] == {}
    store.close()


def test_idempotency_binding_conflict_is_transactional(tmp_path):
    store = OperationStore(tmp_path)
    store.load()
    meta = {"key": ["caller", "image-inspections", "one"], "fingerprint": "hash"}
    store.save({"first": (record("first"), meta)})
    with pytest.raises(ServiceError):
        store.save({"second": (record("second"), meta)})
    assert set(store.load()["records"]) == {"first"}
    store.close()


def test_locked_database_write_fails_without_erasing_committed_state(tmp_path):
    store = OperationStore(tmp_path)
    store.load()
    store.save({"first": (record("first"), {})})
    lock = sqlite3.connect(tmp_path / "operations.sqlite3")
    lock.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(ServiceError) as caught:
            store.save({"second": (record("second"), {})})
        assert caught.value.code == "service.journal_unavailable"
    finally:
        lock.rollback()
        lock.close()
    assert set(store.load()["records"]) == {"first"}
    store.save({"second": (record("second"), {})})
    store.close()


def test_unknown_schema_and_invalid_legacy_journal_are_not_treated_as_empty(tmp_path):
    (tmp_path / "journal.json").write_text("{broken")
    with pytest.raises(ServiceError):
        OperationStore(tmp_path).load()
    (tmp_path / "journal.json").unlink()
    connection = sqlite3.connect(tmp_path / "operations.sqlite3")
    connection.execute("PRAGMA user_version=999")
    connection.close()
    with pytest.raises(ServiceError) as caught:
        OperationStore(tmp_path).load()
    assert caught.value.code == "service.journal_invalid"


def test_export_refuses_active_runtime_owner(tmp_path):
    import fcntl

    with (tmp_path / ".owner.lock").open("a+b") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ServiceError) as caught:
            export_journal(tmp_path, tmp_path / "rollback.json")
    assert caught.value.code == "service.state_in_use"


def test_sqlite_full_rolls_back_only_current_write_and_recovers(tmp_path):
    store = OperationStore(tmp_path)
    store.load()
    store.save({"first": (record("first"), {})})
    pages = store.connection.execute("PRAGMA page_count").fetchone()[0]
    store.connection.execute(f"PRAGMA max_page_count={pages}")
    with pytest.raises(ServiceError) as caught:
        store.save({"large": (record("large", value="x" * 1_000_000), {})})
    assert caught.value.code == "service.journal_unavailable"
    assert set(store.load()["records"]) == {"first"}
    store.connection.execute("PRAGMA max_page_count=2147483646")
    store.save({"second": (record("second"), {})})
    assert set(store.load()["records"]) == {"first", "second"}
    store.close()
