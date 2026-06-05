from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from src.chat_context import chat_db_path, get_active_chat

from dotenv import load_dotenv

from src.telegram_client import ENV_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent



def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(chat_db_path())
    connection.row_factory = sqlite3.Row
    return connection


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        LIMIT 1
        """,
        (table,),
    ).fetchone()

    return row is not None


def get_app_state(
    connection: sqlite3.Connection,
    key: str,
) -> str | None:
    if not table_exists(connection, "app_state"):
        return None

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


def count_rows(
    connection: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...] = (),
) -> int:
    row = connection.execute(query, params).fetchone()
    return int(row[0] or 0)


def print_batches_summary(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "message_batches"):
        print("Batches: table not found")
        return

    rows = connection.execute(
        """
        SELECT direction, status, COUNT(*) AS count
        FROM message_batches
        GROUP BY direction, status
        ORDER BY direction, status
        """
    ).fetchall()

    print()
    print("Batches:")

    if not rows:
        print("  no batches")
        return

    for row in rows:
        print(
            f"  {row['direction']:<9} "
            f"{row['status']:<24} "
            f"{row['count']}"
        )


def print_pending_candidates(connection: sqlite3.Connection) -> None:
    if (
        not table_exists(connection, "message_batches")
        or not table_exists(connection, "candidates")
    ):
        return

    rows = connection.execute(
        """
        SELECT
            mb.batch_id,
            COUNT(c.candidate_id) AS candidates_count,
            ROUND(MAX(c.final_score), 4) AS best_score
        FROM message_batches AS mb
        LEFT JOIN candidates AS c
          ON c.batch_id = mb.batch_id
        WHERE mb.direction = 'incoming'
          AND mb.status = 'candidates_ready'
        GROUP BY mb.batch_id
        ORDER BY mb.batch_id
        """
    ).fetchall()

    print()
    print("Pending candidates:")

    if not rows:
        print("  none")
        return

    for row in rows:
        score = (
            "n/a"
            if row["best_score"] is None
            else row["best_score"]
        )

        print(
            f"  batch #{row['batch_id']}: "
            f"{row['candidates_count']} candidates, "
            f"best_score={score}"
        )


def print_recent_responses(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "responses"):
        return

    rows = connection.execute(
        """
        SELECT
            response_id,
            batch_id,
            source,
            status,
            attribution
        FROM responses
        ORDER BY response_id DESC
        LIMIT 5
        """
    ).fetchall()

    print()
    print("Recent responses:")

    if not rows:
        print("  none")
        return

    for row in rows:
        print(
            f"  response #{row['response_id']} | "
            f"batch #{row['batch_id']} | "
            f"{row['source']} | "
            f"{row['status']} | "
            f"{row['attribution']}"
        )


def print_status() -> None:
    load_dotenv(ENV_PATH)

    db_path = chat_db_path()

    if not db_path.exists():
        print(f"Database not found: {DB_PATH}")
        return

    with connect() as connection:
        mode = (
            get_app_state(connection, "app_mode")
            or os.getenv("APP_MODE", "manual")
        )

        target_user_id = (
            get_app_state(connection, "target_user_id")
            or os.getenv("TARGET_USER_ID", "not set")
        )

        target_chat_id = (
            get_app_state(connection, "target_tdlib_chat_id")
            or os.getenv("TARGET_TDLIB_CHAT_ID", "not set")
        )

        messages_count = (
            count_rows(connection, "SELECT COUNT(*) FROM messages")
            if table_exists(connection, "messages")
            else 0
        )

        print()
        print("Application status")
        print("==================")
        print(f"mode:                 {mode}")
        print(f"target_user_id:       {target_user_id}")
        print(f"target_tdlib_chat_id: {target_chat_id}")
        print(f"messages:             {messages_count}")

        print_batches_summary(connection)
        print_pending_candidates(connection)
        print_recent_responses(connection)
        print()


def main() -> None:
    print_status()


if __name__ == "__main__":
    main()
