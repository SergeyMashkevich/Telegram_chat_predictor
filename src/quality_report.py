from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.chat_context import chat_db_path, get_active_chat


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


def column_exists(
    connection: sqlite3.Connection,
    table: str,
    column: str,
) -> bool:
    rows = connection.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()

    return any(str(row["name"]) == column for row in rows)


def fetch_count(
    connection: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...] = (),
) -> int:
    row = connection.execute(query, params).fetchone()
    return int(row[0] or 0)


def preview_messages(messages_json: str, limit: int = 90) -> str:
    try:
        messages = json.loads(messages_json)
    except json.JSONDecodeError:
        return "[invalid json]"

    text = " / ".join(
        str(message).strip()
        for message in messages
        if str(message).strip()
    )

    if len(text) <= limit:
        return text

    return text[: limit - 3] + "..."


def print_section(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def print_key_values(rows: list[tuple[str, Any]]) -> None:
    width = max((len(str(key)) for key, _ in rows), default=0)

    for key, value in rows:
        print(f"{key:<{width}}  {value}")


def report_messages(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "messages"):
        return

    print_section("Messages")

    total = fetch_count(connection, "SELECT COUNT(*) FROM messages")
    incoming = fetch_count(
        connection,
        "SELECT COUNT(*) FROM messages WHERE is_outgoing = 0",
    )
    outgoing = fetch_count(
        connection,
        "SELECT COUNT(*) FROM messages WHERE is_outgoing = 1",
    )

    print_key_values(
        [
            ("total", total),
            ("incoming", incoming),
            ("outgoing", outgoing),
        ]
    )


def report_batches(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "message_batches"):
        return

    print_section("Message batches")

    rows = connection.execute(
        """
        SELECT direction, status, COUNT(*) AS count
        FROM message_batches
        GROUP BY direction, status
        ORDER BY direction, status
        """
    ).fetchall()

    if not rows:
        print("No batches yet.")
        return

    for row in rows:
        print(
            f"{row['direction']:<10} "
            f"{row['status']:<24} "
            f"{row['count']}"
        )


def report_prediction_runs(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "prediction_runs"):
        return

    print_section("Prediction runs")

    rows = connection.execute(
        """
        SELECT status, COUNT(*) AS count
        FROM prediction_runs
        GROUP BY status
        ORDER BY status
        """
    ).fetchall()

    if not rows:
        print("No prediction runs yet.")
        return

    for row in rows:
        print(f"{row['status']:<14} {row['count']}")


def report_candidates(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "candidates"):
        return

    print_section("Candidates")

    total = fetch_count(connection, "SELECT COUNT(*) FROM candidates")

    rows = connection.execute(
        """
        SELECT position, COUNT(*) AS count
        FROM candidates
        GROUP BY position
        ORDER BY position
        """
    ).fetchall()

    print(f"total: {total}")

    for row in rows:
        print(f"position {row['position']}: {row['count']}")

    if column_exists(connection, "candidates", "final_score"):
        score_row = connection.execute(
            """
            SELECT
                ROUND(AVG(final_score), 4) AS avg_score,
                ROUND(MIN(final_score), 4) AS min_score,
                ROUND(MAX(final_score), 4) AS max_score
            FROM candidates
            WHERE final_score IS NOT NULL
            """
        ).fetchone()

        if score_row and score_row["avg_score"] is not None:
            print()
            print_key_values(
                [
                    ("avg final_score", score_row["avg_score"]),
                    ("min final_score", score_row["min_score"]),
                    ("max final_score", score_row["max_score"]),
                ]
            )


def report_responses(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "responses"):
        return

    print_section("Responses")

    rows = connection.execute(
        """
        SELECT source, status, attribution, COUNT(*) AS count
        FROM responses
        GROUP BY source, status, attribution
        ORDER BY source, status, attribution
        """
    ).fetchall()

    if not rows:
        print("No responses yet.")
        return

    for row in rows:
        print(
            f"{row['source']:<18} "
            f"{row['status']:<12} "
            f"{row['attribution']:<22} "
            f"{row['count']}"
        )


def report_manual_choices(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "responses") or not table_exists(connection, "candidates"):
        return

    print_section("Manual CLI choices")

    rows = connection.execute(
        """
        SELECT
            c.position,
            COUNT(*) AS count,
            ROUND(AVG(c.final_score), 4) AS avg_score
        FROM responses AS r
        JOIN candidates AS c
          ON c.candidate_id = r.candidate_id
        WHERE r.source = 'manual_cli'
          AND r.attribution = 'confirmed'
        GROUP BY c.position
        ORDER BY c.position
        """
    ).fetchall()

    if not rows:
        print("No confirmed manual candidate choices yet.")
        return

    for row in rows:
        score = row["avg_score"] if row["avg_score"] is not None else "n/a"
        print(
            f"candidate [{row['position']}]: "
            f"{row['count']} choices, avg_score={score}"
        )


def report_external_matches(connection: sqlite3.Connection) -> None:
    if not table_exists(connection, "responses"):
        return

    if not column_exists(connection, "responses", "similarity_score"):
        return

    print_section("External Telegram matching")

    rows = connection.execute(
        """
        SELECT
            attribution,
            COUNT(*) AS count,
            ROUND(AVG(similarity_score), 4) AS avg_similarity
        FROM responses
        WHERE source = 'external_telegram'
        GROUP BY attribution
        ORDER BY attribution
        """
    ).fetchall()

    if not rows:
        print("No external Telegram matches yet.")
        return

    for row in rows:
        similarity = (
            row["avg_similarity"]
            if row["avg_similarity"] is not None
            else "n/a"
        )

        print(
            f"{row['attribution']:<22} "
            f"{row['count']} responses, "
            f"avg_similarity={similarity}"
        )

    if column_exists(connection, "responses", "best_matching_candidate_id"):
        print()
        print("Best matching candidate positions:")

        rows = connection.execute(
            """
            SELECT
                c.position,
                COUNT(*) AS count,
                ROUND(AVG(r.similarity_score), 4) AS avg_similarity
            FROM responses AS r
            JOIN candidates AS c
              ON c.candidate_id = r.best_matching_candidate_id
            WHERE r.source = 'external_telegram'
            GROUP BY c.position
            ORDER BY c.position
            """
        ).fetchall()

        if not rows:
            print("No best candidate matches yet.")
        else:
            for row in rows:
                print(
                    f"candidate [{row['position']}]: "
                    f"{row['count']} matches, "
                    f"avg_similarity={row['avg_similarity']}"
                )


def report_recent_responses(connection: sqlite3.Connection, limit: int = 10) -> None:
    if not table_exists(connection, "responses"):
        return

    print_section(f"Recent responses, last {limit}")

    candidate_join = ""
    candidate_select = "NULL AS candidate_position"

    if table_exists(connection, "candidates"):
        candidate_join = """
        LEFT JOIN candidates AS c
          ON c.candidate_id = COALESCE(
              r.candidate_id,
              r.best_matching_candidate_id
          )
        """
        candidate_select = "c.position AS candidate_position"

    rows = connection.execute(
        f"""
        SELECT
            r.response_id,
            r.batch_id,
            r.source,
            r.status,
            r.attribution,
            r.similarity_score,
            {candidate_select},
            r.final_messages_json
        FROM responses AS r
        {candidate_join}
        ORDER BY r.response_id DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()

    if not rows:
        print("No responses yet.")
        return

    for row in rows:
        similarity = row["similarity_score"]

        if similarity is None:
            similarity_text = "n/a"
        else:
            similarity_text = f"{float(similarity):.3f}"

        candidate_position = row["candidate_position"]

        if candidate_position is None:
            candidate_text = "n/a"
        else:
            candidate_text = f"[{candidate_position}]"

        print()
        print(
            f"response #{row['response_id']} | "
            f"batch #{row['batch_id']} | "
            f"{row['source']} | "
            f"{row['status']} | "
            f"candidate={candidate_text} | "
            f"similarity={similarity_text} | "
            f"{row['attribution']}"
        )
        print(preview_messages(str(row["final_messages_json"])))


def main() -> None:
    db_path = chat_db_path()

    if not db_path.exists():
        raise RuntimeError(
            f"Database not found: {db_path}. "
            "Run live_sync first."
        )

    with connect() as connection:
        report_messages(connection)
        report_batches(connection)
        report_prediction_runs(connection)
        report_candidates(connection)
        report_responses(connection)
        report_manual_choices(connection)
        report_external_matches(connection)
        report_recent_responses(connection)


if __name__ == "__main__":
    main()
