"""Transcription history stored in SQLite."""

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    id: int
    timestamp: float
    raw_text: str
    processed_text: str
    app_context: str
    mode: str  # "dictation", "command", "question"
    language: str
    duration: float


class TranscriptionHistory:
    """Stores transcription history in a local SQLite database."""

    def __init__(self, path: str, max_entries: int = 5000):
        self.path = path
        self.max_entries = max_entries
        self._init_db()

    def _init_db(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    raw_text TEXT NOT NULL,
                    processed_text TEXT NOT NULL,
                    app_context TEXT DEFAULT '',
                    mode TEXT DEFAULT 'dictation',
                    language TEXT DEFAULT '',
                    duration REAL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON history(timestamp)
            """)

    def add(
        self,
        raw_text: str,
        processed_text: str,
        app_context: str = "",
        mode: str = "dictation",
        language: str = "",
        duration: float = 0.0,
    ) -> int:
        """Add a transcription to history. Returns the entry ID."""
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute(
                """INSERT INTO history (timestamp, raw_text, processed_text, app_context, mode, language, duration)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (time.time(), raw_text, processed_text, app_context, mode, language, duration),
            )
            entry_id = cursor.lastrowid

            # Prune old entries
            conn.execute(
                """DELETE FROM history WHERE id NOT IN
                   (SELECT id FROM history ORDER BY timestamp DESC LIMIT ?)""",
                (self.max_entries,),
            )

            return entry_id

    def get_recent(self, limit: int = 50) -> list[HistoryEntry]:
        """Get recent history entries."""
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """SELECT id, timestamp, raw_text, processed_text, app_context, mode, language, duration
                   FROM history ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()

        return [HistoryEntry(*row) for row in rows]

    def search(self, query: str, limit: int = 20) -> list[HistoryEntry]:
        """Search history by text content."""
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """SELECT id, timestamp, raw_text, processed_text, app_context, mode, language, duration
                   FROM history WHERE processed_text LIKE ? OR raw_text LIKE ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()

        return [HistoryEntry(*row) for row in rows]

    def get_entry(self, entry_id: int) -> HistoryEntry | None:
        """Get a specific history entry."""
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                """SELECT id, timestamp, raw_text, processed_text, app_context, mode, language, duration
                   FROM history WHERE id = ?""",
                (entry_id,),
            ).fetchone()

        return HistoryEntry(*row) if row else None

    def delete_entry(self, entry_id: int) -> None:
        """Delete a history entry."""
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM history WHERE id = ?", (entry_id,))

    def clear(self) -> None:
        """Clear all history."""
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM history")

    @property
    def count(self) -> int:
        with sqlite3.connect(self.path) as conn:
            return conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
