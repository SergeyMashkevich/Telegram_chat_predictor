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


def initialize_response_matching_storage() -> None:
    with connect() as connection:
        _ensure_column(
            connection,
            "responses",
            "best_matching_candidate_id",
            "INTEGER",
        )
        _ensure_column(
            connection,
            "responses",
            "similarity_score",
            "REAL",
        )
        _ensure_column(
            connection,
            "responses",
            "matching_status",
            "TEXT NOT NULL DEFAULT 'pending'",
        )
        _ensure_column(
            connection,
            "responses",
            "matched_at",
            "INTEGER",
        )


def claim_next_response_for_matching() -> dict[str, Any] | None:
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        row = connection.execute(
            """
            SELECT
                response_id,
                batch_id,
                final_messages_json
            FROM responses
            WHERE source = 'external_telegram'
              AND candidate_id IS NULL
              AND status = 'observed'
              AND matching_status IN ('pending', 'failed')
              AND EXISTS (
                  SELECT 1
                  FROM candidates
                  WHERE candidates.batch_id = responses.batch_id
              )
            ORDER BY response_id
            LIMIT 1
            """
        ).fetchone()

        if row is None:
            return None

        response_id = int(row["response_id"])

        connection.execute(
            """
            UPDATE responses
            SET matching_status = 'matching'
            WHERE response_id = ?
              AND matching_status IN ('pending', 'failed')
            """,
            (response_id,),
        )

    return {
        "response_id": response_id,
        "batch_id": int(row["batch_id"]),
        "final_messages": json.loads(
            str(row["final_messages_json"])
        ),
    }


def get_candidates_for_response_batch(
    batch_id: int,
) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT
                candidate_id,
                position,
                messages_json
            FROM candidates
            WHERE batch_id = ?
            ORDER BY position
            """,
            (int(batch_id),),
        ).fetchall()

    result: list[dict[str, Any]] = []

    for row in rows:
        result.append(
            {
                "candidate_id": int(row["candidate_id"]),
                "position": int(row["position"]),
                "messages": json.loads(
                    str(row["messages_json"])
                ),
            }
        )

    return result


def save_response_match(
    response_id: int,
    candidate_id: int | None,
    similarity_score: float | None,
    attribution: str,
) -> None:
    now = int(time.time())

    with connect() as connection:
        connection.execute(
            """
            UPDATE responses
            SET best_matching_candidate_id = ?,
                similarity_score = ?,
                attribution = ?,
                matching_status = 'matched',
                matched_at = ?
            WHERE response_id = ?
            """,
            (
                candidate_id,
                similarity_score,
                attribution,
                now,
                int(response_id),
            ),
        )


def mark_response_matching_failed(
    response_id: int,
    error_text: str,
) -> None:
    with connect() as connection:
        connection.execute(
            """
            UPDATE responses
            SET matching_status = 'failed',
                error_text = ?
            WHERE response_id = ?
            """,
            (
                error_text[:4000],
                int(response_id),
            ),
        )
