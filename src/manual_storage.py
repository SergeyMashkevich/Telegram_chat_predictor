from __future__ import annotations

import json
import time
from typing import Any

from src.prediction_storage import initialize_prediction_storage
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


def initialize_manual_storage() -> None:
    initialize_prediction_storage()

    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS responses (
                response_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL UNIQUE,
                candidate_id INTEGER,
                source TEXT NOT NULL,
                attribution TEXT NOT NULL,
                final_messages_json TEXT NOT NULL,
                reply_to_message_id INTEGER,
                reply_used INTEGER,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                sent_at INTEGER,
                error_text TEXT,
                FOREIGN KEY (batch_id)
                    REFERENCES message_batches(batch_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (candidate_id)
                    REFERENCES candidates(candidate_id)
                    ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS response_messages (
                response_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                PRIMARY KEY (chat_id, message_id),
                UNIQUE (response_id, position),
                FOREIGN KEY (response_id)
                    REFERENCES responses(response_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_responses_status
            ON responses (status);

            CREATE INDEX IF NOT EXISTS idx_response_messages_response
            ON response_messages (response_id, position);
            """
        )

        _ensure_column(
            connection,
            "responses",
            "reply_to_message_id",
            "INTEGER",
        )
        _ensure_column(
            connection,
            "responses",
            "reply_used",
            "INTEGER",
        )


def get_oldest_pending_batch_id(chat_id: int) -> int | None:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT batch_id
            FROM message_batches
            WHERE chat_id = ?
              AND direction = 'incoming'
              AND status = 'candidates_ready'
            ORDER BY batch_id
            LIMIT 1
            """,
            (int(chat_id),),
        ).fetchone()

    if row is None:
        return None

    return int(row["batch_id"])


def get_reply_target_message_id(
    batch_id: int,
    reply_to_incoming_index: int,
) -> int | None:
    if reply_to_incoming_index <= 0:
        return None

    with connect() as connection:
        row = connection.execute(
            """
            SELECT message_id
            FROM batch_messages
            WHERE batch_id = ?
            ORDER BY position
            LIMIT 1 OFFSET ?
            """,
            (
                int(batch_id),
                int(reply_to_incoming_index) - 1,
            ),
        ).fetchone()

    if row is None:
        return None

    return int(row["message_id"])


def get_pending_candidates(
    chat_id: int,
) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT
                mb.batch_id,
                c.candidate_id,
                c.position,
                c.messages_json,
                c.reply_to_incoming_index
            FROM message_batches AS mb
            JOIN candidates AS c
              ON c.batch_id = mb.batch_id
            WHERE mb.chat_id = ?
              AND mb.direction = 'incoming'
              AND mb.status = 'candidates_ready'
            ORDER BY mb.batch_id, c.position
            """,
            (int(chat_id),),
        ).fetchall()

    batches: dict[int, dict[str, Any]] = {}

    for row in rows:
        batch_id = int(row["batch_id"])

        batch = batches.setdefault(
            batch_id,
            {
                "batch_id": batch_id,
                "candidates": [],
            },
        )

        batch["candidates"].append(
            {
                "candidate_id": int(row["candidate_id"]),
                "position": int(row["position"]),
                "messages": json.loads(
                    str(row["messages_json"])
                ),
                "reply_to_incoming_index": int(
                    row["reply_to_incoming_index"] or 0
                ),
            }
        )

    return list(batches.values())


def claim_candidate_for_send(
    batch_id: int,
    position: int,
) -> dict[str, Any] | None:
    now = int(time.time())

    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        batch = connection.execute(
            """
            SELECT chat_id, status
            FROM message_batches
            WHERE batch_id = ?
            """,
            (int(batch_id),),
        ).fetchone()

        if batch is None or batch["status"] != "candidates_ready":
            return None

        candidate = connection.execute(
            """
            SELECT
                candidate_id,
                messages_json,
                reply_to_incoming_index
            FROM candidates
            WHERE batch_id = ?
              AND position = ?
            """,
            (
                int(batch_id),
                int(position),
            ),
        ).fetchone()

        if candidate is None:
            return None

        cursor = connection.execute(
            """
            UPDATE message_batches
            SET status = 'manual_send_queued'
            WHERE batch_id = ?
              AND status = 'candidates_ready'
            """,
            (int(batch_id),),
        )

        if cursor.rowcount != 1:
            return None

        messages_json = str(candidate["messages_json"])
        reply_index = int(candidate["reply_to_incoming_index"] or 0)

        reply_to_message_id = get_reply_target_message_id(
            batch_id=batch_id,
            reply_to_incoming_index=reply_index,
        )

        response_cursor = connection.execute(
            """
            INSERT INTO responses (
                batch_id,
                candidate_id,
                source,
                attribution,
                final_messages_json,
                reply_to_message_id,
                reply_used,
                status,
                created_at
            )
            VALUES (?, ?, 'manual_cli', 'confirmed', ?, ?, NULL, 'queued', ?)
            """,
            (
                int(batch_id),
                int(candidate["candidate_id"]),
                messages_json,
                reply_to_message_id,
                now,
            ),
        )

        response_id = int(response_cursor.lastrowid)

    return {
        "response_id": response_id,
        "batch_id": int(batch_id),
        "chat_id": int(batch["chat_id"]),
        "messages": json.loads(messages_json),
        "reply_to_message_id": reply_to_message_id,
    }


def claim_custom_response_for_send(
    batch_id: int,
    text: str,
) -> dict[str, Any] | None:
    clean_text = text.strip()

    if not clean_text:
        return None

    now = int(time.time())
    messages = [clean_text]
    messages_json = json.dumps(messages, ensure_ascii=False)

    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        batch = connection.execute(
            """
            SELECT chat_id, status
            FROM message_batches
            WHERE batch_id = ?
            """,
            (int(batch_id),),
        ).fetchone()

        if batch is None or batch["status"] != "candidates_ready":
            return None

        cursor = connection.execute(
            """
            UPDATE message_batches
            SET status = 'manual_send_queued'
            WHERE batch_id = ?
              AND status = 'candidates_ready'
            """,
            (int(batch_id),),
        )

        if cursor.rowcount != 1:
            return None

        response_cursor = connection.execute(
            """
            INSERT INTO responses (
                batch_id,
                candidate_id,
                source,
                attribution,
                final_messages_json,
                reply_to_message_id,
                reply_used,
                status,
                created_at
            )
            VALUES (?, NULL, 'manual_cli', 'custom', ?, NULL, NULL, 'queued', ?)
            """,
            (
                int(batch_id),
                messages_json,
                now,
            ),
        )

        response_id = int(response_cursor.lastrowid)

    return {
        "response_id": response_id,
        "batch_id": int(batch_id),
        "chat_id": int(batch["chat_id"]),
        "messages": messages,
        "reply_to_message_id": None,
    }


def skip_batch(batch_id: int) -> bool:
    now = int(time.time())

    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        cursor = connection.execute(
            """
            UPDATE message_batches
            SET status = 'skipped',
                closed_at = COALESCE(closed_at, ?)
            WHERE batch_id = ?
              AND status = 'candidates_ready'
            """,
            (
                now,
                int(batch_id),
            ),
        )

        if cursor.rowcount != 1:
            return False

        connection.execute(
            """
            INSERT INTO responses (
                batch_id,
                candidate_id,
                source,
                attribution,
                final_messages_json,
                reply_to_message_id,
                reply_used,
                status,
                created_at
            )
            VALUES (?, NULL, 'manual_cli', 'skip', '[]', NULL, NULL, 'skipped', ?)
            """,
            (
                int(batch_id),
                now,
            ),
        )

    return True


def _save_response_message_ids(
    connection: Any,
    response_id: int,
    chat_id: int,
    message_ids: list[int],
) -> None:
    connection.executemany(
        """
        INSERT INTO response_messages (
            response_id,
            chat_id,
            message_id,
            position
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(chat_id, message_id) DO NOTHING
        """,
        [
            (
                int(response_id),
                int(chat_id),
                int(message_id),
                position,
            )
            for position, message_id in enumerate(
                message_ids,
                start=1,
            )
        ],
    )


def mark_response_sent(
    response_id: int,
    chat_id: int,
    message_ids: list[int],
    reply_used: bool,
) -> None:
    now = int(time.time())

    with connect() as connection:
        _save_response_message_ids(
            connection=connection,
            response_id=response_id,
            chat_id=chat_id,
            message_ids=message_ids,
        )

        row = connection.execute(
            """
            SELECT batch_id
            FROM responses
            WHERE response_id = ?
            """,
            (int(response_id),),
        ).fetchone()

        if row is None:
            return

        batch_id = int(row["batch_id"])

        connection.execute(
            """
            UPDATE responses
            SET status = 'sent',
                sent_at = ?,
                reply_used = ?,
                error_text = NULL
            WHERE response_id = ?
            """,
            (
                now,
                int(bool(reply_used)),
                int(response_id),
            ),
        )

        connection.execute(
            """
            UPDATE message_batches
            SET status = CASE
                WHEN status = 'manual_send_queued' THEN 'manual_sent'
                WHEN status = 'auto_send_queued' THEN 'auto_sent'
                ELSE status
            END
            WHERE batch_id = ?
              AND status IN ('manual_send_queued', 'auto_send_queued')
            """,
            (batch_id,),
        )


def mark_response_failed(
    response_id: int,
    chat_id: int,
    message_ids: list[int],
    error_text: str,
) -> None:
    with connect() as connection:
        _save_response_message_ids(
            connection=connection,
            response_id=response_id,
            chat_id=chat_id,
            message_ids=message_ids,
        )

        row = connection.execute(
            """
            SELECT batch_id
            FROM responses
            WHERE response_id = ?
            """,
            (int(response_id),),
        ).fetchone()

        if row is None:
            return

        batch_id = int(row["batch_id"])

        connection.execute(
            """
            UPDATE responses
            SET status = 'failed',
                error_text = ?
            WHERE response_id = ?
            """,
            (
                error_text[:4000],
                int(response_id),
            ),
        )

        connection.execute(
            """
            UPDATE message_batches
            SET status = CASE
                WHEN status = 'manual_send_queued' THEN 'manual_send_failed'
                WHEN status = 'auto_send_queued' THEN 'auto_send_failed'
                ELSE status
            END
            WHERE batch_id = ?
              AND status IN ('manual_send_queued', 'auto_send_queued')
            """,
            (batch_id,),
        )


def is_application_sent_message(
    chat_id: int,
    message_id: int,
) -> bool:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM response_messages
            WHERE chat_id = ?
              AND message_id = ?
            LIMIT 1
            """,
            (
                int(chat_id),
                int(message_id),
            ),
        ).fetchone()

    return row is not None


def replace_response_message_id(
    chat_id: int,
    old_message_id: int,
    new_message_id: int,
) -> None:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT response_id, position
            FROM response_messages
            WHERE chat_id = ?
              AND message_id = ?
            """,
            (
                int(chat_id),
                int(old_message_id),
            ),
        ).fetchone()

        if row is None:
            return

        connection.execute(
            """
            DELETE FROM response_messages
            WHERE chat_id = ?
              AND message_id = ?
            """,
            (
                int(chat_id),
                int(old_message_id),
            ),
        )

        connection.execute(
            """
            INSERT INTO response_messages (
                response_id,
                chat_id,
                message_id,
                position
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id, message_id) DO NOTHING
            """,
            (
                int(row["response_id"]),
                int(chat_id),
                int(new_message_id),
                int(row["position"]),
            ),
        )


def record_external_telegram_response(
    outgoing_batch_id: int,
) -> int | None:
    now = int(time.time())

    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        outgoing_batch = connection.execute(
            """
            SELECT *
            FROM message_batches
            WHERE batch_id = ?
              AND direction = 'outgoing'
              AND status = 'outgoing_ready'
            """,
            (int(outgoing_batch_id),),
        ).fetchone()

        if outgoing_batch is None:
            return None

        outgoing_messages_rows = connection.execute(
            """
            SELECT
                bm.message_id,
                bm.position,
                m.raw_json
            FROM batch_messages AS bm
            JOIN messages AS m
              ON m.chat_id = bm.chat_id
             AND m.message_id = bm.message_id
            WHERE bm.batch_id = ?
            ORDER BY bm.position
            """,
            (int(outgoing_batch_id),),
        ).fetchall()

        if not outgoing_messages_rows:
            connection.execute(
                """
                UPDATE message_batches
                SET status = 'outgoing_empty'
                WHERE batch_id = ?
                """,
                (int(outgoing_batch_id),),
            )
            return None

        outgoing_messages = [
            json.loads(str(row["raw_json"]))
            for row in outgoing_messages_rows
        ]

        final_messages = [
            str(message.get("text", "")).strip()
            for message in outgoing_messages
            if str(message.get("text", "")).strip()
        ]

        if not final_messages:
            connection.execute(
                """
                UPDATE message_batches
                SET status = 'outgoing_empty'
                WHERE batch_id = ?
                """,
                (int(outgoing_batch_id),),
            )
            return None

        chat_id = int(outgoing_batch["chat_id"])
        first_outgoing_message_id = int(outgoing_batch["first_message_id"])

        incoming_batch = connection.execute(
            """
            SELECT mb.batch_id
            FROM message_batches AS mb
            LEFT JOIN responses AS r
              ON r.batch_id = mb.batch_id
            WHERE mb.chat_id = ?
              AND mb.direction = 'incoming'
              AND mb.status = 'answered_manually'
              AND mb.last_message_id < ?
              AND r.response_id IS NULL
            ORDER BY mb.batch_id DESC
            LIMIT 1
            """,
            (
                chat_id,
                first_outgoing_message_id,
            ),
        ).fetchone()

        if incoming_batch is None:
            connection.execute(
                """
                UPDATE message_batches
                SET status = 'outgoing_unlinked'
                WHERE batch_id = ?
                """,
                (int(outgoing_batch_id),),
            )
            return None

        incoming_batch_id = int(incoming_batch["batch_id"])
        final_messages_json = json.dumps(
            final_messages,
            ensure_ascii=False,
        )

        response_cursor = connection.execute(
            """
            INSERT INTO responses (
                batch_id,
                candidate_id,
                source,
                attribution,
                final_messages_json,
                reply_to_message_id,
                reply_used,
                status,
                created_at,
                sent_at
            )
            VALUES (
                ?,
                NULL,
                'external_telegram',
                'unknown',
                ?,
                NULL,
                NULL,
                'observed',
                ?,
                ?
            )
            """,
            (
                incoming_batch_id,
                final_messages_json,
                now,
                now,
            ),
        )

        response_id = int(response_cursor.lastrowid)

        connection.executemany(
            """
            INSERT INTO response_messages (
                response_id,
                chat_id,
                message_id,
                position
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chat_id, message_id) DO NOTHING
            """,
            [
                (
                    response_id,
                    chat_id,
                    int(row["message_id"]),
                    int(row["position"]) + 1,
                )
                for row in outgoing_messages_rows
            ],
        )

        connection.execute(
            """
            UPDATE message_batches
            SET status = 'external_answer_recorded'
            WHERE batch_id = ?
            """,
            (incoming_batch_id,),
        )

        connection.execute(
            """
            UPDATE message_batches
            SET status = 'outgoing_recorded'
            WHERE batch_id = ?
            """,
            (int(outgoing_batch_id),),
        )

    return incoming_batch_id


def claim_best_candidate_for_auto_send(
    chat_id: int,
    min_final_score: float,
    delay_seconds: float,
) -> dict[str, Any] | None:
    now = int(time.time())
    cutoff_time = now - int(delay_seconds)

    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        batch = connection.execute(
            """
            SELECT *
            FROM message_batches
            WHERE chat_id = ?
              AND direction = 'incoming'
              AND status = 'candidates_ready'
              AND COALESCE(closed_at, last_activity_at, created_at) <= ?
            ORDER BY batch_id
            LIMIT 1
            """,
            (
                int(chat_id),
                int(cutoff_time),
            ),
        ).fetchone()

        if batch is None:
            return None

        batch_id = int(batch["batch_id"])

        candidate = connection.execute(
            """
            SELECT
                candidate_id,
                position,
                messages_json,
                reply_to_incoming_index,
                final_score
            FROM candidates
            WHERE batch_id = ?
              AND (
                  final_score >= ?
                  OR (? <= 0 AND final_score IS NULL)
              )
            ORDER BY
                COALESCE(final_score, -1.0) DESC,
                position ASC
            LIMIT 1
            """,
            (
                batch_id,
                float(min_final_score),
                float(min_final_score),
            ),
        ).fetchone()

        if candidate is None:
            return None

        cursor = connection.execute(
            """
            UPDATE message_batches
            SET status = 'auto_send_queued'
            WHERE batch_id = ?
              AND status = 'candidates_ready'
            """,
            (batch_id,),
        )

        if cursor.rowcount != 1:
            return None

        messages_json = str(candidate["messages_json"])
        reply_index = int(candidate["reply_to_incoming_index"] or 0)

        reply_to_message_id = get_reply_target_message_id(
            batch_id=batch_id,
            reply_to_incoming_index=reply_index,
        )

        response_cursor = connection.execute(
            """
            INSERT INTO responses (
                batch_id,
                candidate_id,
                source,
                attribution,
                final_messages_json,
                reply_to_message_id,
                reply_used,
                status,
                created_at
            )
            VALUES (?, ?, 'auto', 'auto', ?, ?, NULL, 'queued', ?)
            """,
            (
                batch_id,
                int(candidate["candidate_id"]),
                messages_json,
                reply_to_message_id,
                now,
            ),
        )

        response_id = int(response_cursor.lastrowid)

    return {
        "response_id": response_id,
        "batch_id": batch_id,
        "chat_id": int(batch["chat_id"]),
        "candidate_position": int(candidate["position"]),
        "final_score": candidate["final_score"],
        "messages": json.loads(messages_json),
        "reply_to_message_id": reply_to_message_id,
    }
