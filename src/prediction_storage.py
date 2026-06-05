from __future__ import annotations

import json
import time
from typing import Any

from src.storage import connect


def _ensure_column(
    connection: Any,
    table_name: str,
    column_name: str,
    definition: str,
) -> None:
    columns = connection.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    existing = {str(column["name"]) for column in columns}

    if column_name not in existing:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
        )


def initialize_prediction_storage() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS prediction_runs (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL UNIQUE,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at INTEGER NOT NULL,
                completed_at INTEGER,
                error_text TEXT,
                raw_response TEXT,
                FOREIGN KEY (batch_id)
                    REFERENCES message_batches(batch_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                messages_json TEXT NOT NULL,
                reply_to_incoming_index INTEGER NOT NULL DEFAULT 0,
                logprob_score REAL,
                style_score REAL,
                relevance_score REAL,
                final_score REAL,
                created_at INTEGER NOT NULL,
                UNIQUE (batch_id, position),
                FOREIGN KEY (batch_id)
                    REFERENCES message_batches(batch_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_candidates_batch
            ON candidates (batch_id, position);
            """
        )

        _ensure_column(
            connection,
            "candidates",
            "reply_to_incoming_index",
            "INTEGER NOT NULL DEFAULT 0",
        )


def recover_stuck_generating_batches(chat_id: int) -> None:
    now = int(time.time())

    with connect() as connection:
        connection.execute(
            """
            UPDATE prediction_runs
            SET status = 'interrupted',
                completed_at = ?
            WHERE status = 'generating'
              AND batch_id IN (
                  SELECT batch_id
                  FROM message_batches
                  WHERE chat_id = ?
                    AND status = 'generating'
              )
            """,
            (now, int(chat_id)),
        )

        connection.execute(
            """
            UPDATE message_batches
            SET status = 'ready_for_prediction'
            WHERE chat_id = ?
              AND status = 'generating'
            """,
            (int(chat_id),),
        )


def claim_next_ready_batch(
    chat_id: int,
    model: str,
) -> int | None:
    now = int(time.time())

    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        row = connection.execute(
            """
            SELECT batch_id
            FROM message_batches
            WHERE chat_id = ?
              AND direction = 'incoming'
              AND status = 'ready_for_prediction'
            ORDER BY batch_id
            LIMIT 1
            """,
            (int(chat_id),),
        ).fetchone()

        if row is None:
            return None

        batch_id = int(row["batch_id"])

        cursor = connection.execute(
            """
            UPDATE message_batches
            SET status = 'generating'
            WHERE batch_id = ?
              AND status = 'ready_for_prediction'
            """,
            (batch_id,),
        )

        if cursor.rowcount != 1:
            return None

        connection.execute(
            """
            INSERT INTO prediction_runs (
                batch_id,
                model,
                status,
                started_at,
                completed_at,
                error_text,
                raw_response
            )
            VALUES (?, ?, 'generating', ?, NULL, NULL, NULL)
            ON CONFLICT(batch_id) DO UPDATE SET
                model = excluded.model,
                status = 'generating',
                started_at = excluded.started_at,
                completed_at = NULL,
                error_text = NULL,
                raw_response = NULL
            """,
            (
                batch_id,
                model,
                now,
            ),
        )

    return batch_id


def get_batch_info(batch_id: int) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM message_batches
            WHERE batch_id = ?
            """,
            (int(batch_id),),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


def get_recent_messages_before(
    chat_id: int,
    before_message_id: int,
    limit: int,
) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT raw_json
            FROM messages
            WHERE chat_id = ?
              AND message_id < ?
            ORDER BY message_id DESC
            LIMIT ?
            """,
            (
                int(chat_id),
                int(before_message_id),
                int(limit),
            ),
        ).fetchall()

    messages = [
        json.loads(str(row["raw_json"]))
        for row in rows
    ]

    messages.reverse()

    return messages


def _normalize_candidate(
    candidate: Any,
) -> tuple[list[str], int]:
    if isinstance(candidate, dict):
        raw_messages = candidate.get("messages", [])
        reply_index = int(candidate.get("reply_to_incoming_index", 0))
    else:
        raw_messages = candidate
        reply_index = 0

    messages = [
        str(message).strip()
        for message in raw_messages
        if str(message).strip()
    ]

    if reply_index < 0:
        reply_index = 0

    return messages, reply_index


def save_candidates_if_generating(
    batch_id: int,
    candidates: list[Any],
    raw_response: str,
) -> bool:
    now = int(time.time())

    normalized_candidates = [
        _normalize_candidate(candidate)
        for candidate in candidates
    ]

    normalized_candidates = [
        (messages, reply_index)
        for messages, reply_index in normalized_candidates
        if messages
    ]

    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        row = connection.execute(
            """
            SELECT status
            FROM message_batches
            WHERE batch_id = ?
            """,
            (int(batch_id),),
        ).fetchone()

        if row is None or row["status"] != "generating":
            return False

        connection.execute(
            """
            DELETE FROM candidates
            WHERE batch_id = ?
            """,
            (int(batch_id),),
        )

        connection.executemany(
            """
            INSERT INTO candidates (
                batch_id,
                position,
                messages_json,
                reply_to_incoming_index,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    int(batch_id),
                    position,
                    json.dumps(messages, ensure_ascii=False),
                    int(reply_index),
                    now,
                )
                for position, (messages, reply_index) in enumerate(
                    normalized_candidates,
                    start=1,
                )
            ],
        )

        connection.execute(
            """
            UPDATE message_batches
            SET status = 'candidates_ready'
            WHERE batch_id = ?
              AND status = 'generating'
            """,
            (int(batch_id),),
        )

        connection.execute(
            """
            UPDATE prediction_runs
            SET status = 'completed',
                completed_at = ?,
                raw_response = ?
            WHERE batch_id = ?
            """,
            (
                now,
                raw_response,
                int(batch_id),
            ),
        )

    return True


def mark_generation_failed(
    batch_id: int,
    error_text: str,
) -> None:
    now = int(time.time())

    with connect() as connection:
        connection.execute(
            """
            UPDATE message_batches
            SET status = 'generation_failed'
            WHERE batch_id = ?
              AND status = 'generating'
            """,
            (int(batch_id),),
        )

        connection.execute(
            """
            UPDATE prediction_runs
            SET status = 'failed',
                completed_at = ?,
                error_text = ?
            WHERE batch_id = ?
            """,
            (
                now,
                error_text[:4000],
                int(batch_id),
            ),
        )


def get_candidates_for_batch(
    batch_id: int,
) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM candidates
            WHERE batch_id = ?
            ORDER BY position
            """,
            (int(batch_id),),
        ).fetchall()

    candidates: list[dict[str, Any]] = []

    for row in rows:
        candidate = dict(row)
        candidate["messages"] = json.loads(
            str(candidate.pop("messages_json"))
        )
        candidates.append(candidate)

    return candidates


def ensure_candidate_score_columns() -> None:
    with connect() as connection:
        _ensure_column(
            connection,
            "candidates",
            "logprob_score",
            "REAL",
        )
        _ensure_column(
            connection,
            "candidates",
            "style_score",
            "REAL",
        )
        _ensure_column(
            connection,
            "candidates",
            "relevance_score",
            "REAL",
        )
        _ensure_column(
            connection,
            "candidates",
            "final_score",
            "REAL",
        )


def update_candidate_scores(
    batch_id: int,
    ranked_candidates: list[dict[str, Any]],
) -> None:
    ensure_candidate_score_columns()

    with connect() as connection:
        for position, candidate in enumerate(ranked_candidates, start=1):
            connection.execute(
                """
                UPDATE candidates
                SET style_score = ?,
                    relevance_score = ?,
                    final_score = ?
                WHERE batch_id = ?
                  AND position = ?
                """,
                (
                    candidate.get("style_score"),
                    candidate.get("relevance_score"),
                    candidate.get("final_score"),
                    int(batch_id),
                    int(position),
                ),
            )
