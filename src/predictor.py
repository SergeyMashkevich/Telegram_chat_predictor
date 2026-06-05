from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from ollama import Client

from src.telegram_client import ENV_PATH


def require_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(f"Переменная {name} не указана в .env")

    return value.strip()


def format_reactions(message: dict[str, Any]) -> str:
    reactions = message.get("reactions", [])

    if not reactions:
        return ""

    labels: list[str] = []

    for reaction in reactions:
        reaction_type = reaction.get("type")

        if reaction_type == "emoji":
            label = str(reaction.get("emoji", "")).strip() or "[emoji]"
        elif reaction_type == "custom_emoji":
            label = "[custom emoji]"
        elif reaction_type == "paid":
            label = "[paid reaction]"
        else:
            label = "[reaction]"

        count = int(reaction.get("total_count", 0))
        labels.append(f"{label} × {count}")

    return f" [реакции: {', '.join(labels)}]"


def render_message(message: dict[str, Any]) -> str:
    role = "Вы" if message.get("is_outgoing") else "Собеседник"
    text = str(message.get("text", "")).strip()

    return f"{role}: {text}{format_reactions(message)}"


def render_transcript(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "(история отсутствует)"

    return "\n".join(render_message(message) for message in messages)


def render_numbered_incoming(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "(входящий блок отсутствует)"

    return "\n".join(
        f"[{index}] {render_message(message)}"
        for index, message in enumerate(messages, start=1)
    )


class OllamaPredictor:
    def __init__(self) -> None:
        load_dotenv(ENV_PATH)

        self.model = require_env("OLLAMA_CHAT_MODEL")
        self.host = require_env("OLLAMA_HOST")

        self.candidate_count = int(os.getenv("CANDIDATE_COUNT", "3"))
        self.max_messages_per_candidate = int(
            os.getenv("MAX_MESSAGES_PER_CANDIDATE", "4")
        )
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

        self.client = Client(host=self.host)

    def candidate_schema(self, incoming_message_count: int) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": self.candidate_count,
                    "maxItems": self.candidate_count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": self.max_messages_per_candidate,
                                "items": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                            },
                            "reply_to_incoming_index": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": incoming_message_count,
                            },
                        },
                        "required": [
                            "messages",
                            "reply_to_incoming_index",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["candidates"],
            "additionalProperties": False,
        }

    def parse_candidate_specs(
        self,
        content: str,
        incoming_message_count: int,
    ) -> list[dict[str, Any]]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise RuntimeError("Ollama вернула некорректный JSON.") from error

        raw_candidates = payload.get("candidates", [])

        if not isinstance(raw_candidates, list):
            raise RuntimeError("Поле candidates в ответе Ollama не является массивом.")

        candidates: list[dict[str, Any]] = []
        seen_messages: set[str] = set()

        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                continue

            raw_messages = raw_candidate.get("messages", [])

            if not isinstance(raw_messages, list):
                continue

            messages = [
                str(message).strip()
                for message in raw_messages
                if str(message).strip()
            ]

            if not messages:
                continue

            if len(messages) > self.max_messages_per_candidate:
                continue

            reply_index = raw_candidate.get("reply_to_incoming_index", 0)

            if type(reply_index) is not int:
                continue

            if not 0 <= reply_index <= incoming_message_count:
                continue

            messages_key = json.dumps(messages, ensure_ascii=False)

            if messages_key in seen_messages:
                continue

            seen_messages.add(messages_key)

            candidates.append(
                {
                    "messages": messages,
                    "reply_to_incoming_index": reply_index,
                }
            )

        return self._cap_reply_recommendations(candidates)

    def _cap_reply_recommendations(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Reply не должен появляться у всех кандидатов.
        Даже если модель рекомендует reply слишком часто, оставляем reply
        максимум у половины кандидатов, но не больше чем у одного при 3 вариантах.
        """
        reply_candidates = [
            candidate
            for candidate in candidates
            if int(candidate.get("reply_to_incoming_index", 0)) > 0
        ]

        if not reply_candidates:
            return candidates

        max_reply_candidates = max(1, len(candidates) // 2)

        allowed_reply_indexes = set(range(max_reply_candidates))
        seen_reply_index = 0

        normalized: list[dict[str, Any]] = []

        for candidate in candidates:
            candidate = dict(candidate)

            if int(candidate.get("reply_to_incoming_index", 0)) > 0:
                if seen_reply_index not in allowed_reply_indexes:
                    candidate["reply_to_incoming_index"] = 0

                seen_reply_index += 1

            normalized.append(candidate)

        return normalized

    def has_message_count_diversity(
        self,
        candidates: list[dict[str, Any]],
    ) -> bool:
        if len(candidates) <= 1:
            return True

        counts = {len(candidate["messages"]) for candidate in candidates}

        return len(counts) >= 2

    def generate_candidate_specs(
        self,
        history_messages: list[dict[str, Any]],
        incoming_messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        incoming_message_count = len(incoming_messages)

        if incoming_message_count == 0:
            raise RuntimeError("Нельзя генерировать ответ для пустого входящего блока.")

        system_prompt = f"""
Вы предсказываете следующее сообщение, которое пользователь написал бы
в личной переписке Telegram.

Главный приоритет: ответ должен быть осмысленной реакцией на текущий
входящий блок. Стиль пользователя важен, но нельзя жертвовать смыслом.

Имитируйте стиль пользователя только по предыдущим сообщениям,
помеченным словом "Вы".

Не отвечайте как помощник.
Не объясняйте свой выбор.
Не пишите бессмысленные слова.
Не используйте случайные слоги, опечатки или странные слова, если они
не требуются текущим контекстом.
Не делайте сообщения формальнее, чем стиль пользователя.
Не придумывайте факты, которых нет в переписке.
Учитывайте эмодзи, стикеры, медиа-placeholder и реакции.

Сформируйте ровно {self.candidate_count} разных вероятных вариантов ответа.

Каждый вариант может состоять от 1 до
{self.max_messages_per_candidate} отдельных Telegram-сообщений.
Выбирайте количество сообщений естественно.
Короткий ответ обычно должен состоять из одного сообщения.
Если мысль естественно разбивается, можно использовать несколько сообщений.

Для каждого кандидата укажите reply_to_incoming_index:
- 0, если reply не нужен;
- номер входящего сообщения в квадратных скобках, если первое исходящее
  сообщение действительно естественно отправить как reply на него.

Reply используйте редко. Не ставьте reply у всех кандидатов.
Reply относится только к первому сообщению кандидата.

Текст сообщений должен быть без кавычек, нумерации и Markdown.
""".strip()

        base_user_prompt = f"""
Предыдущая переписка:

{render_transcript(history_messages)}

Текущий входящий блок, на который нужно ответить:

{render_numbered_incoming(incoming_messages)}
""".strip()

        last_error = ""
        last_valid_candidates: list[dict[str, Any]] = []
        last_raw_response = ""

        for attempt in range(1, 4):
            user_prompt = base_user_prompt

            if attempt > 1:
                user_prompt += (
                    "\n\nПредыдущая попытка была недостаточно естественной. "
                    "Сделайте ответы более осмысленными, без случайных слов. "
                    "Не используйте reply у всех кандидатов."
                )

            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                format=self.candidate_schema(
                    incoming_message_count=incoming_message_count,
                ),
                stream=False,
                think=False,
                keep_alive=self.keep_alive,
                options={
                    "temperature": 0.65,
                },
            )

            content = response.message.content

            candidates = self.parse_candidate_specs(
                content=content,
                incoming_message_count=incoming_message_count,
            )

            if len(candidates) == self.candidate_count:
                last_valid_candidates = candidates
                last_raw_response = content

                if self.has_message_count_diversity(candidates):
                    return candidates, content

                last_error = "Все кандидаты содержат одинаковое количество сообщений."
                continue

            last_error = (
                "Ollama вернула "
                f"{len(candidates)} уникальных кандидатов "
                f"вместо {self.candidate_count}."
            )

        if last_valid_candidates:
            return last_valid_candidates, last_raw_response

        raise RuntimeError(last_error)

    def generate_candidates(
        self,
        history_messages: list[dict[str, Any]],
        incoming_messages: list[dict[str, Any]],
    ) -> tuple[list[list[str]], str]:
        candidate_specs, raw_response = self.generate_candidate_specs(
            history_messages=history_messages,
            incoming_messages=incoming_messages,
        )

        candidates = [
            list(candidate["messages"])
            for candidate in candidate_specs
        ]

        return candidates, raw_response
