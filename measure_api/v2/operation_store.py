"""Incremental operation state with one SQLite writer and reversible journal migration."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Dict, Iterable, Optional, Tuple

from measure_api.v2.errors import ServiceError

SCHEMA_VERSION = 1


class OperationStore:
    """Own one connection used exclusively by the runtime's serial I/O executor."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.connection: Optional[sqlite3.Connection] = None

    def load(self) -> dict:
        """Load retained state, importing a legacy journal only into a new database."""
        try:
            if self.connection is None:
                self.directory.mkdir(parents=True, exist_ok=True)
                self.connection = sqlite3.connect(
                    self.directory / "operations.sqlite3", timeout=0.2)
                self.connection.execute("PRAGMA journal_mode=WAL")
                self.connection.execute("PRAGMA synchronous=FULL")
                version = self.connection.execute("PRAGMA user_version").fetchone()[0]
                if version == 0:
                    self._initialize()
                elif version != SCHEMA_VERSION:
                    raise ValueError("Unsupported operational state schema")
            saved = {"records": {}, "metadata": {}, "expired": {}, "expired_traces": {}}
            for identity, document, metadata in self.connection.execute(
                    "SELECT id,document,metadata FROM operations"):
                saved["records"][identity] = json.loads(document)
                saved["metadata"][identity] = json.loads(metadata)
            for kind, identity, at in self.connection.execute(
                    "SELECT kind,identity,expired_at FROM expiration_markers"):
                key = "expired" if kind == "operation" else "expired_traces"
                saved[key][identity] = at
            return saved
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as error:
            self.close()
            raise ServiceError("service.journal_invalid", "Operational state is unreadable",
                               503) from error

    def _initialize(self) -> None:
        saved = {"records": {}, "metadata": {}, "expired": {}, "expired_traces": {}}
        journal = self.directory / "journal.json"
        if journal.exists():
            saved.update(json.loads(journal.read_text()))
        for name in ("records", "metadata", "expired", "expired_traces"):
            if not isinstance(saved[name], dict):
                raise ValueError("Invalid legacy operational journal")
        connection = self.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""CREATE TABLE operations (
                id TEXT PRIMARY KEY, document TEXT NOT NULL, metadata TEXT NOT NULL,
                caller TEXT, route TEXT, retry_key TEXT, fingerprint TEXT,
                UNIQUE(caller,route,retry_key))""")
            connection.execute("""CREATE TABLE expiration_markers (
                kind TEXT NOT NULL, identity TEXT NOT NULL, expired_at REAL NOT NULL,
                PRIMARY KEY(kind,identity))""")
            for identity, record in saved["records"].items():
                self._upsert(identity, json.dumps(record, allow_nan=False),
                             saved["metadata"].get(identity, {"terminal_at": None}))
            for name, kind in (("expired", "operation"), ("expired_traces", "trace")):
                connection.executemany("INSERT INTO expiration_markers VALUES(?,?,?)",
                                       [(kind, identity, at)
                                        for identity, at in saved[name].items()])
            connection.execute("PRAGMA user_version=1")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def _upsert(self, identity: str, document: str, metadata: dict) -> None:
        key = metadata.get("key") or (None, None, None)
        self.connection.execute("""INSERT INTO operations VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET document=excluded.document,
                metadata=excluded.metadata,caller=excluded.caller,route=excluded.route,
                retry_key=excluded.retry_key,fingerprint=excluded.fingerprint""",
            (identity, document, json.dumps(metadata, allow_nan=False),
             *key, metadata.get("fingerprint")))

    def save(self, updates: Dict[str, Tuple[dict, dict]], removed: Iterable[str] = (),
             markers: Optional[Dict[Tuple[str, str], Optional[float]]] = None) -> dict:
        """Commit only supplied records and expiration changes as one durable transaction."""
        started = time.perf_counter()
        try:
            rows = [(identity, json.dumps(record, allow_nan=False, separators=(",", ":")), meta)
                    for identity, (record, meta) in updates.items()]
            encoded = time.perf_counter()
            self._write(rows, removed, markers or {})
            return {"encode_ms": (encoded - started) * 1000,
                    "commit_ms": (time.perf_counter() - encoded) * 1000,
                    "encoded_bytes": sum(len(row[1].encode()) for row in rows),
                    "records": len(rows)}
        except (OSError, sqlite3.Error, ValueError, TypeError) as error:
            raise ServiceError("service.journal_unavailable",
                               "Operational state cannot be written", 503,
                               retryable=True) from error

    def _write(self, rows: list, removed: Iterable[str], markers: dict) -> None:
        with self.connection:
            self.connection.executemany("DELETE FROM operations WHERE id=?",
                                        [(identity,) for identity in removed])
            for identity, document, meta in rows:
                self._upsert(identity, document, meta)
            for (kind, identity), at in markers.items():
                if at is None:
                    self.connection.execute(
                        "DELETE FROM expiration_markers WHERE kind=? AND identity=?",
                        (kind, identity))
                else:
                    self.connection.execute("""INSERT INTO expiration_markers VALUES(?,?,?)
                        ON CONFLICT(kind,identity) DO UPDATE SET expired_at=excluded.expired_at""",
                        (kind, identity, at))

    def close(self) -> None:
        """Close the writer connection on the same thread that opened it."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def export_journal(directory: Path, output: Path) -> None:
    """Export a consistent legacy rollback journal while the runtime owner is stopped."""
    directory, output = Path(directory), Path(output)
    with (directory / ".owner.lock").open("a+b") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ServiceError("service.state_in_use", "Stop the runtime before export", 503
                               ) from error
        store = OperationStore(directory)
        try:
            saved = store.load()
            descriptor, stage = tempfile.mkstemp(prefix=".journal-", dir=output.parent)
            try:
                with os.fdopen(descriptor, "w") as target:
                    json.dump(saved, target, allow_nan=False)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(stage, output)
                directory_fd = os.open(output.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                Path(stage).unlink(missing_ok=True)
        finally:
            store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export"])
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_journal(args.state_directory, args.output)
