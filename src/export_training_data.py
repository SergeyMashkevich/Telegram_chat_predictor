from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.chat_context import get_active_chat
from src.storage import connect
from src.telegram_client import ENV_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


URL_RE = re.compile(
    r"""(?ix)
    \b
    (
        https?://\S+
        |
        www\.\S+
        |
        t\.me/\S+
        |
        telegram\.me/\S+
    )
    """
)


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = URL_RE.sub("[ссылка]", text)
    return " ".join(text.split())


def join_block(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    for message in messages:
        text = clean_text(str(message.get("text", "")))

        if text:
            lines.append(text)

    return "\n".join(lines)


def render_context(messages: list[dict[str, Any]]) -> str:
    rendered: list[str] = []

    for message in messages:
        role = "Вы" if int(message["is_outgoing"]) else "Собеседник"
        text = clean_text(str(message["text"]))

        if text:
            rendered.append(f"{role}: {text}")

    return "\n".join(rendered)


def load_messages() -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT
                chat_id,
                message_id,
                is_outgoing,
                date_unixtime,
                text,
                raw_json
            FROM messages
            WHERE text IS NOT NULL
              AND TRIM(text) != ''
            ORDER BY date_unixtime ASC, message_id ASC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def split_into_direction_blocks(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    current_direction: str | None = None
    current_messages: list[dict[str, Any]] = []

    for message in messages:
        direction = "outgoing" if int(message["is_outgoing"]) else "incoming"

        if current_direction is None:
            current_direction = direction
            current_messages = [message]
            continue

        if direction == current_direction:
            current_messages.append(message)
            continue

        blocks.append(
            {
                "direction": current_direction,
                "messages": current_messages,
            }
        )

        current_direction = direction
        current_messages = [message]

    if current_messages and current_direction is not None:
        blocks.append(
            {
                "direction": current_direction,
                "messages": current_messages,
            }
        )

    return blocks


def build_examples(
    blocks: list[dict[str, Any]],
    context_blocks: int,
    min_target_chars: int,
    max_target_chars: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for index, block in enumerate(blocks):
        if block["direction"] != "outgoing":
            continue

        if index == 0:
            continue

        previous_block = blocks[index - 1]

        if previous_block["direction"] != "incoming":
            continue

        target = join_block(block["messages"])

        if len(target) < min_target_chars:
            continue

        if len(target) > max_target_chars:
            continue

        context_start = max(0, index - context_blocks)
        context_blocks_slice = blocks[context_start:index]

        context_messages: list[dict[str, Any]] = []

        for context_block in context_blocks_slice:
            context_messages.extend(context_block["messages"])

        incoming_text = join_block(previous_block["messages"])

        if not incoming_text:
            continue

        context_text = render_context(context_messages)

        user_prompt = f"""
Предыдущая переписка:

{context_text}

Текущий входящий блок, на который нужно ответить:

{incoming_text}
""".strip()

        examples.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Имитируй стиль пользователя в личной переписке Telegram. "
                            "Ответь так, как пользователь вероятнее всего ответил бы сам. "
                            "Не объясняй ответ."
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": target,
                    },
                ],
                "metadata": {
                    "target_message_ids": [
                        int(message["message_id"])
                        for message in block["messages"]
                    ],
                    "incoming_message_ids": [
                        int(message["message_id"])
                        for message in previous_block["messages"]
                    ],
                    "target_message_count": len(block["messages"]),
                    "target_chars": len(target),
                },
            }
        )

    return examples


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv(ENV_PATH)

    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "Чат не выбран. Сначала выполните make run или make select-chat."
        )

    context_blocks = int(os.getenv("TRAIN_CONTEXT_BLOCKS", "8"))
    min_target_chars = int(os.getenv("TRAIN_MIN_TARGET_CHARS", "1"))
    max_target_chars = int(os.getenv("TRAIN_MAX_TARGET_CHARS", "800"))

    messages = load_messages()
    blocks = split_into_direction_blocks(messages)

    examples = build_examples(
        blocks=blocks,
        context_blocks=context_blocks,
        min_target_chars=min_target_chars,
        max_target_chars=max_target_chars,
    )

    chat_id = int(active_chat["chat_id"])
    output_path = DATASETS_DIR / "chats" / str(chat_id) / "sft.jsonl"

    write_jsonl(output_path, examples)

    print()
    print("Training dataset exported")
    print("=========================")
    print(f"chat:       {active_chat['title']}")
    print(f"chat_id:    {chat_id}")
    print(f"messages:   {len(messages)}")
    print(f"blocks:     {len(blocks)}")
    print(f"examples:   {len(examples)}")
    print(f"output:     {output_path}")

    if examples:
        print()
        print("First example preview")
        print("=====================")
        print("USER:")
        print(examples[0]["messages"][1]["content"][:1200])
        print()
        print("ASSISTANT:")
        print(examples[0]["messages"][2]["content"][:600])


if __name__ == "__main__":
    main()
