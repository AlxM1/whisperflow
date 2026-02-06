"""Transcription history stored in SQLite."""

import csv
import io
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    id: int
    timestamp: float
    raw_text: str
    processed_text: str
    app_context: str
    mode: str  # "dictation", "command", "question", "transcribe"
    language: str
    duration: float
    audio_path: str = ""


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
                    duration REAL DEFAULT 0.0,
                    audio_path TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON history(timestamp)
            """)
            # Migrate: add audio_path column if missing (for existing DBs)
            cols = {row[1] for row in conn.execute("PRAGMA table_info(history)").fetchall()}
            if "audio_path" not in cols:
                conn.execute("ALTER TABLE history ADD COLUMN audio_path TEXT DEFAULT ''")

    def add(
        self,
        raw_text: str,
        processed_text: str,
        app_context: str = "",
        mode: str = "dictation",
        language: str = "",
        duration: float = 0.0,
        audio_path: str = "",
    ) -> int:
        """Add a transcription to history. Returns the entry ID."""
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute(
                """INSERT INTO history (timestamp, raw_text, processed_text, app_context, mode, language, duration, audio_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (time.time(), raw_text, processed_text, app_context, mode, language, duration, audio_path),
            )
            entry_id = cursor.lastrowid

            # Prune old entries
            conn.execute(
                """DELETE FROM history WHERE id NOT IN
                   (SELECT id FROM history ORDER BY timestamp DESC LIMIT ?)""",
                (self.max_entries,),
            )

            return entry_id

    _SELECT_COLS = "id, timestamp, raw_text, processed_text, app_context, mode, language, duration, audio_path"

    def get_recent(self, limit: int = 50) -> list[HistoryEntry]:
        """Get recent history entries."""
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                f"SELECT {self._SELECT_COLS} FROM history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [HistoryEntry(*row) for row in rows]

    def search(self, query: str, limit: int = 20) -> list[HistoryEntry]:
        """Search history by text content."""
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                f"""SELECT {self._SELECT_COLS} FROM history
                    WHERE processed_text LIKE ? OR raw_text LIKE ?
                    ORDER BY timestamp DESC LIMIT ?""",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [HistoryEntry(*row) for row in rows]

    def get_entry(self, entry_id: int) -> HistoryEntry | None:
        """Get a specific history entry."""
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                f"SELECT {self._SELECT_COLS} FROM history WHERE id = ?",
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

    def get_stats(self) -> dict:
        """Get usage statistics."""
        with sqlite3.connect(self.path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
            total_duration = conn.execute(
                "SELECT COALESCE(SUM(duration), 0) FROM history"
            ).fetchone()[0]
            total_words = 0
            for row in conn.execute("SELECT processed_text FROM history").fetchall():
                total_words += len(row[0].split())
            modes = {}
            for row in conn.execute(
                "SELECT mode, COUNT(*) FROM history GROUP BY mode"
            ).fetchall():
                modes[row[0] or "dictation"] = row[1]
            languages = {}
            for row in conn.execute(
                "SELECT language, COUNT(*) FROM history WHERE language != '' GROUP BY language ORDER BY COUNT(*) DESC LIMIT 10"
            ).fetchall():
                languages[row[0]] = row[1]
            # Recent activity (last 7 days)
            week_ago = time.time() - 7 * 86400
            recent_count = conn.execute(
                "SELECT COUNT(*) FROM history WHERE timestamp > ?", (week_ago,)
            ).fetchone()[0]
            recent_duration = conn.execute(
                "SELECT COALESCE(SUM(duration), 0) FROM history WHERE timestamp > ?", (week_ago,)
            ).fetchone()[0]
        return {
            "total_transcriptions": total,
            "total_duration_seconds": round(total_duration, 1),
            "total_words": total_words,
            "modes": modes,
            "languages": languages,
            "last_7_days": {"transcriptions": recent_count, "duration_seconds": round(recent_duration, 1)},
        }

    def export_csv(self) -> str:
        """Export all history as CSV string."""
        entries = self.get_recent(limit=self.max_entries)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "timestamp", "raw_text", "processed_text", "app_context", "mode", "language", "duration"])
        for e in entries:
            writer.writerow([e.id, e.timestamp, e.raw_text, e.processed_text, e.app_context, e.mode, e.language, e.duration])
        return output.getvalue()

    def export_json(self) -> str:
        """Export all history as JSON string."""
        entries = self.get_recent(limit=self.max_entries)
        return json.dumps([asdict(e) for e in entries], indent=2, ensure_ascii=False)
