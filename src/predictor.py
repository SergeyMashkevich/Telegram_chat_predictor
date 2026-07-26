from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from ollama import Client

from src.telegram_client import ENV_PATH
from src.model_registry import resolve_chat_model


def require_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(f"Variable {name} is not set in .env")

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

    return f" [reactions: {', '.join(labels)}]"


def render_message(message: dict[str, Any]) -> str:
    role = "You" if message.get("is_outgoing") else "Chat partner"
    text = str(message.get("text", "")).strip()

    return f"{role}: {text}{format_reactions(message)}"


def render_transcript(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "(no previous conversation)"

    return "\n".join(render_message(message) for message in messages)


def render_numbered_incoming(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "(no incoming batch)"

    return "\n".join(
        f"[{index}] {render_message(message)}"
        for index, message in enumerate(messages, start=1)
    )


class OllamaPredictor:
    def __init__(self) -> None:
        load_dotenv(ENV_PATH)

        self.model = resolve_chat_model()
        self.host = require_env("OLLAMA_HOST")

        self.candidate_count = int(os.getenv("CANDIDATE_COUNT", "3"))
        self.max_messages_per_candidate = int(
            os.getenv("MAX_MESSAGES_PER_CANDIDATE", "4")
        )
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

        self.client = Client(host=self.host)

    def refresh_model(self) -> str:
        self.model = resolve_chat_model()
        return self.model

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
            raise RuntimeError("Ollama returned invalid JSON.") from error

        raw_candidates = payload.get("candidates", [])

        if not isinstance(raw_candidates, list):
            raise RuntimeError(
                "The candidates field in the Ollama response is not an array."
            )

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
        A reply should not appear on every candidate.
        Even if the model recommends replies too often, keep replies on at most
        half of the candidates, and on no more than one when there are 3 options.
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
        self.refresh_model()

        incoming_message_count = len(incoming_messages)

        if incoming_message_count == 0:
            raise RuntimeError(
                "Cannot generate a response for an empty incoming batch."
            )

        system_prompt = f"""
Predict the next message the user would write in a private Telegram chat.

The highest priority is a meaningful response to the current incoming batch.
The user's style matters, but meaning must not be sacrificed.

Imitate the user's style using only previous messages labeled "You".
Use the language naturally implied by the conversation.

Do not answer as an assistant.
Do not explain your choices.
Do not write meaningless words.
Do not use random syllables, typos, or unusual words unless the current
context calls for them.
Do not make messages more formal than the user's style.
Do not invent facts that are not present in the conversation.
Account for emoji, stickers, media placeholders, and reactions.

Generate exactly {self.candidate_count} distinct, plausible response options.

Each option may contain between 1 and
{self.max_messages_per_candidate} separate Telegram messages.
Choose the number of messages naturally.
A short response should usually contain one message.
Use multiple messages when the thought naturally splits into several parts.

For each candidate, provide reply_to_incoming_index:
- 0 when no reply is needed;
- the incoming message number shown in square brackets when the first outgoing
  message would naturally be sent as a reply to it.

Use replies sparingly. Do not add a reply to every candidate.
A reply applies only to the candidate's first message.

Message text must not include quotation marks, numbering, or Markdown.
""".strip()

        base_user_prompt = f"""
Previous conversation:

{render_transcript(history_messages)}

Current incoming batch to answer:

{render_numbered_incoming(incoming_messages)}
""".strip()

        last_error = ""
        last_valid_candidates: list[dict[str, Any]] = []
        last_raw_response = ""

        for attempt in range(1, 4):
            user_prompt = base_user_prompt

            if attempt > 1:
                user_prompt += (
                    "\n\nThe previous attempt was not natural enough. "
                    "Make the responses more meaningful and avoid random words. "
                    "Do not use a reply on every candidate."
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

                last_error = "All candidates contain the same number of messages."
                continue

            last_error = (
                "Ollama returned "
                f"{len(candidates)} unique candidates "
                f"instead of {self.candidate_count}."
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
