from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from src.chat_context import chat_db_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent



def connect() -> sqlite3.Connection:
    db_path = chat_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")

    return connection


def initialize_database() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                chat_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                sender_id TEXT,
                sender_name TEXT,
                is_outgoing INTEGER NOT NULL,
                date_unixtime INTEGER NOT NULL,
                text TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                PRIMARY KEY (chat_id, message_id)
            );

            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS message_batches (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                direction TEXT NOT NULL,
                status TEXT NOT NULL,
                first_message_id INTEGER NOT NULL,
                last_message_id INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                last_activity_at INTEGER NOT NULL,
                closed_at INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_message_batches_lookup
            ON message_batches (
                chat_id,
                direction,
                status,
                last_activity_at
            );

            CREATE TABLE IF NOT EXISTS batch_messages (
                batch_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                PRIMARY KEY (batch_id, message_id),
                UNIQUE (chat_id, message_id),
                FOREIGN KEY (batch_id)
                    REFERENCES message_batches(batch_id)
                    ON DELETE CASCADE
            );
            """
        )


def upsert_messages(messages: Iterable[dict[str, Any]]) -> None:
    rows = []

    for message in messages:
        rows.append(
            (
                int(message["tdlib_chat_id"]),
                int(message["id"]),
                message.get("from_id"),
                message.get("from"),
                int(bool(message.get("is_outgoing"))),
                int(message["date_unixtime"]),
                str(message["text"]),
                json.dumps(message, ensure_ascii=False),
            )
        )

    if not rows:
        return

    with connect() as connection:
        connection.executemany(
            """
            INSERT INTO messages (
                chat_id,
                message_id,
                sender_id,
                sender_name,
                is_outgoing,
                date_unixtime,
                text,
                raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id, message_id) DO UPDATE SET
                sender_id = excluded.sender_id,
                sender_name = excluded.sender_name,
                is_outgoing = excluded.is_outgoing,
                date_unixtime = excluded.date_unixtime,
                text = excluded.text,
                raw_json = excluded.raw_json
            """,
            rows,
        )


def get_message(
    chat_id: int,
    message_id: int,
) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT raw_json
            FROM messages
            WHERE chat_id = ?
              AND message_id = ?
            """,
            (int(chat_id), int(message_id)),
        ).fetchone()

    if row is None:
        return None

    return json.loads(str(row["raw_json"]))


def delete_messages(chat_id: int, message_ids: Iterable[int]) -> None:
    ids = [int(message_id) for message_id in message_ids]

    if not ids:
        return

    placeholders = ", ".join("?" for _ in ids)
    parameters = [int(chat_id), *ids]

    with connect() as connection:
        connection.execute(
            f"""
            DELETE FROM batch_messages
            WHERE chat_id = ?
              AND message_id IN ({placeholders})
            """,
            parameters,
        )

        connection.execute(
            f"""
            DELETE FROM messages
            WHERE chat_id = ?
              AND message_id IN ({placeholders})
            """,
            parameters,
        )


def set_app_state(key: str, value: str) -> None:
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO app_state (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value
            """,
            (key, value),
        )


def get_app_state(key: str) -> str | None:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT value
            FROM app_state
            WHERE key = ?
            """,
            (key,),
        ).fetchone()

    if row is None:
        return None

    return str(row["value"])


def message_has_batch(chat_id: int, message_id: int) -> bool:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM batch_messages
            WHERE chat_id = ?
              AND message_id = ?
            LIMIT 1
            """,
            (int(chat_id), int(message_id)),
        ).fetchone()

    return row is not None


def get_collecting_batch(
    chat_id: int,
    direction: str,
) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM message_batches
            WHERE chat_id = ?
              AND direction = ?
              AND status = 'collecting'
            ORDER BY batch_id DESC
            LIMIT 1
            """,
            (int(chat_id), direction),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


def create_message_batch(
    chat_id: int,
    direction: str,
    message_id: int,
    activity_time: int | None = None,
) -> int:
    now = int(activity_time or time.time())

    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO message_batches (
                chat_id,
                direction,
                status,
                first_message_id,
                last_message_id,
                created_at,
                last_activity_at,
                closed_at
            )
            VALUES (?, ?, 'collecting', ?, ?, ?, ?, NULL)
            """,
            (
                int(chat_id),
                direction,
                int(message_id),
                int(message_id),
                now,
                now,
            ),
        )

        batch_id = int(cursor.lastrowid)

        connection.execute(
            """
            INSERT INTO batch_messages (
                batch_id,
                chat_id,
                message_id,
                position
            )
            VALUES (?, ?, ?, 0)
            """,
            (
                batch_id,
                int(chat_id),
                int(message_id),
            ),
        )

    return batch_id


def append_message_to_batch(
    batch_id: int,
    chat_id: int,
    message_id: int,
    activity_time: int | None = None,
) -> bool:
    now = int(activity_time or time.time())

    with connect() as connection:
        already_exists = connection.execute(
            """
            SELECT 1
            FROM batch_messages
            WHERE chat_id = ?
              AND message_id = ?
            LIMIT 1
            """,
            (int(chat_id), int(message_id)),
        ).fetchone()

        if already_exists is not None:
            return False

        row = connection.execute(
            """
            SELECT COALESCE(MAX(position), -1) + 1 AS next_position
            FROM batch_messages
            WHERE batch_id = ?
            """,
            (int(batch_id),),
        ).fetchone()

        next_position = int(row["next_position"])

        connection.execute(
            """
            INSERT INTO batch_messages (
                batch_id,
                chat_id,
                message_id,
                position
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                int(batch_id),
                int(chat_id),
                int(message_id),
                next_position,
            ),
        )

        connection.execute(
            """
            UPDATE message_batches
            SET last_message_id = ?,
                last_activity_at = ?
            WHERE batch_id = ?
            """,
            (
                int(message_id),
                now,
                int(batch_id),
            ),
        )

    return True


def close_batch(
    batch_id: int,
    status: str,
) -> None:
    now = int(time.time())

    with connect() as connection:
        connection.execute(
            """
            UPDATE message_batches
            SET status = ?,
                closed_at = ?
            WHERE batch_id = ?
              AND status = 'collecting'
            """,
            (
                status,
                now,
                int(batch_id),
            ),
        )


def close_due_incoming_batches(
    chat_id: int,
    cutoff_time: int,
) -> list[int]:
    now = int(time.time())

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT batch_id
            FROM message_batches
            WHERE chat_id = ?
              AND direction = 'incoming'
              AND status = 'collecting'
              AND last_activity_at <= ?
            ORDER BY batch_id
            """,
            (
                int(chat_id),
                int(cutoff_time),
            ),
        ).fetchall()

        batch_ids = [int(row["batch_id"]) for row in rows]

        if batch_ids:
            placeholders = ", ".join("?" for _ in batch_ids)

            connection.execute(
                f"""
                UPDATE message_batches
                SET status = 'ready_for_prediction',
                    closed_at = ?
                WHERE batch_id IN ({placeholders})
                """,
                [now, *batch_ids],
            )

    return batch_ids


def mark_latest_incoming_batch_answered(
    chat_id: int,
) -> int | None:
    now = int(time.time())

    with connect() as connection:
        row = connection.execute(
            """
            SELECT batch_id
            FROM message_batches
            WHERE chat_id = ?
              AND direction = 'incoming'
              AND status IN (
                  'collecting',
                  'ready_for_prediction',
                  'generating',
                  'candidates_ready',
                  'generation_failed'
              )
            ORDER BY batch_id DESC
            LIMIT 1
            """,
            (int(chat_id),),
        ).fetchone()

        if row is None:
            return None

        batch_id = int(row["batch_id"])

        connection.execute(
            """
            UPDATE message_batches
            SET status = 'answered_manually',
                closed_at = ?
            WHERE batch_id = ?
            """,
            (
                now,
                batch_id,
            ),
        )

    return batch_id


def get_batch_messages(batch_id: int) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT m.raw_json
            FROM batch_messages AS bm
            JOIN messages AS m
              ON m.chat_id = bm.chat_id
             AND m.message_id = bm.message_id
            WHERE bm.batch_id = ?
            ORDER BY bm.position
            """,
            (int(batch_id),),
        ).fetchall()

    return [
        json.loads(str(row["raw_json"]))
        for row in rows
    ]


def replace_message_id_in_batches(
    chat_id: int,
    old_message_id: int,
    new_message_id: int,
) -> None:
    with connect() as connection:
        old_row = connection.execute(
            """
            SELECT batch_id, position
            FROM batch_messages
            WHERE chat_id = ?
              AND message_id = ?
            """,
            (
                int(chat_id),
                int(old_message_id),
            ),
        ).fetchone()

        if old_row is None:
            return

        existing_new = connection.execute(
            """
            SELECT 1
            FROM batch_messages
            WHERE chat_id = ?
              AND message_id = ?
            LIMIT 1
            """,
            (
                int(chat_id),
                int(new_message_id),
            ),
        ).fetchone()

        if existing_new is not None:
            connection.execute(
                """
                DELETE FROM batch_messages
                WHERE chat_id = ?
                  AND message_id = ?
                """,
                (
                    int(chat_id),
                    int(old_message_id),
                ),
            )
        else:
            connection.execute(
                """
                UPDATE batch_messages
                SET message_id = ?
                WHERE chat_id = ?
                  AND message_id = ?
                """,
                (
                    int(new_message_id),
                    int(chat_id),
                    int(old_message_id),
                ),
            )

        connection.execute(
            """
            UPDATE message_batches
            SET first_message_id = CASE
                    WHEN first_message_id = ? THEN ?
                    ELSE first_message_id
                END,
                last_message_id = CASE
                    WHEN last_message_id = ? THEN ?
                    ELSE last_message_id
                END
            WHERE chat_id = ?
            """,
            (
                int(old_message_id),
                int(new_message_id),
                int(old_message_id),
                int(new_message_id),
                int(chat_id),
            ),
        )


def close_due_outgoing_batches(
    chat_id: int,
    cutoff_time: int,
) -> list[int]:
    now = int(time.time())

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT batch_id
            FROM message_batches
            WHERE chat_id = ?
              AND direction = 'outgoing'
              AND status = 'collecting'
              AND last_activity_at <= ?
            ORDER BY batch_id
            """,
            (
                int(chat_id),
                int(cutoff_time),
            ),
        ).fetchall()

        batch_ids = [int(row["batch_id"]) for row in rows]

        if batch_ids:
            placeholders = ", ".join("?" for _ in batch_ids)

            connection.execute(
                f"""
                UPDATE message_batches
                SET status = 'outgoing_ready',
                    closed_at = ?
                WHERE batch_id IN ({placeholders})
                """,
                [now, *batch_ids],
            )

    return batch_ids
