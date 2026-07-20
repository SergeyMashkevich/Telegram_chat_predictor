from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src import active_model_metadata
from src.chat_context import get_active_chat
from src.telegram_client import ENV_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "adapters"


def message_text(message: dict[str, Any]) -> str:
    return str(message.get("text", "")).strip()


def render_message(message: dict[str, Any]) -> str:
    role = "Вы" if bool(message.get("is_outgoing")) else "Собеседник"
    text = message_text(message)

    return f"{role}: {text}"


def render_transcript(messages: list[dict[str, Any]]) -> str:
    rendered = [
        render_message(message)
        for message in messages
        if message_text(message)
    ]

    if not rendered:
        return "(история отсутствует)"

    return "\n".join(rendered)


def render_numbered_incoming(messages: list[dict[str, Any]]) -> str:
    rendered = []

    for index, message in enumerate(messages, start=1):
        text = message_text(message)

        if text:
            rendered.append(f"[{index}] Собеседник: {text}")

    if not rendered:
        return "(входящий блок отсутствует)"

    return "\n".join(rendered)


def active_adapter_dir() -> Path:
    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "Чат не выбран. Сначала выполните make run или make select-chat."
        )

    return ADAPTERS_DIR / "chats" / str(active_chat["chat_id"]) / "lora"


def extract_generated_text(output: str) -> str:
    parts = output.split("==========")

    if len(parts) >= 3:
        return parts[1].strip()

    lines = []

    for line in output.splitlines():
        if line.startswith("Prompt:"):
            break
        if line.startswith("Generation:"):
            break
        if line.startswith("Peak memory:"):
            break

        lines.append(line)

    return "\n".join(lines).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(
            "MLX-модель не вернула JSON-объект."
        )

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"MLX-модель вернула некорректный JSON: {candidate[:500]}"
        ) from error


class MLXPredictor:
    def __init__(self) -> None:
        load_dotenv(ENV_PATH)

        self.model = self._resolve_model()

        self.candidate_count = int(os.getenv("CANDIDATE_COUNT", "3"))
        self.max_messages_per_candidate = int(
            os.getenv("MAX_MESSAGES_PER_CANDIDATE", "4")
        )

        self.max_tokens = os.getenv("MLX_MAX_TOKENS", "700")
        self.temperature = os.getenv("MLX_TEMPERATURE", "0.55")

        self.adapter_path = active_adapter_dir()

        if not (self.adapter_path / "adapters.safetensors").exists():
            raise RuntimeError(
                f"MLX adapter не найден: {self.adapter_path / 'adapters.safetensors'}"
            )

    def _resolve_model(self) -> str:
        return active_model_metadata.resolve_active_mlx_base_model()

    def refresh_model(self) -> str:
        load_dotenv(ENV_PATH)

        self.model = self._resolve_model()
        self.adapter_path = active_adapter_dir()

        return self.model

    def build_prompt(
        self,
        history_messages: list[dict[str, Any]],
        incoming_messages: list[dict[str, Any]],
        attempt: int,
    ) -> str:
        system_prompt = (
            "Ты предсказываешь следующее сообщение, которое пользователь "
            "написал бы в личной переписке Telegram. "
            "Имитируй только стиль пользователя, помеченного как «Вы». "
            "Ответ должен быть естественным, коротким и уместным. "
            "Не объясняй ответ. "
            "Верни только JSON без Markdown."
        )

        user_prompt = f"""
Предыдущая переписка:

{render_transcript(history_messages)}

Текущий входящий блок, на который нужно ответить:

{render_numbered_incoming(incoming_messages)}

Сформируй ровно {self.candidate_count} разных кандидатов.

Формат ответа строго такой:
{{
  "candidates": [
    {{
      "messages": ["текст первого Telegram-сообщения"],
      "reply_to_incoming_index": 0
    }}
  ]
}}

Правила:
- candidates должен содержать ровно {self.candidate_count} элементов.
- messages содержит от 1 до {self.max_messages_per_candidate} сообщений.
- reply_to_incoming_index = 0, если reply не нужен.
- reply_to_incoming_index = номер входящего сообщения, если reply естественен.
- Не используй Markdown.
- Не добавляй текст вне JSON.
""".strip()

        if attempt > 1:
            user_prompt += (
                "\n\nПредыдущая попытка была некорректной. "
                "Верни только валидный JSON, без комментариев и без Markdown."
            )

        if "qwen" in self.model.lower():
            return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
""".strip()

        return f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""".strip()

    def run_generate(self, prompt: str) -> str:
        command = [
            "mlx_lm.generate",
            "--model",
            self.model,
            "--adapter-path",
            str(self.adapter_path),
            "--prompt",
            prompt,
            "--max-tokens",
            self.max_tokens,
            "--temp",
            self.temperature,
        ]

        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        return extract_generated_text(completed.stdout)

    def parse_candidate_specs(
        self,
        payload: dict[str, Any],
        incoming_message_count: int,
    ) -> list[dict[str, Any]]:
        raw_candidates = payload.get("candidates", [])

        if not isinstance(raw_candidates, list):
            raise RuntimeError("Поле candidates не является массивом.")

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

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
                reply_index = 0

            if not 0 <= reply_index <= incoming_message_count:
                reply_index = 0

            key = json.dumps(messages, ensure_ascii=False)

            if key in seen:
                continue

            seen.add(key)

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
        max_reply_candidates = max(1, len(candidates) // 2)
        seen_reply = 0
        normalized: list[dict[str, Any]] = []

        for candidate in candidates:
            candidate = dict(candidate)

            if int(candidate.get("reply_to_incoming_index", 0)) > 0:
                if seen_reply >= max_reply_candidates:
                    candidate["reply_to_incoming_index"] = 0

                seen_reply += 1

            normalized.append(candidate)

        return normalized

    def generate_candidate_specs(
        self,
        history_messages: list[dict[str, Any]],
        incoming_messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        self.refresh_model()

        incoming_message_count = len(incoming_messages)

        if incoming_message_count == 0:
            raise RuntimeError("Нельзя генерировать ответ для пустого блока.")

        last_error = ""

        for attempt in range(1, 4):
            prompt = self.build_prompt(
                history_messages=history_messages,
                incoming_messages=incoming_messages,
                attempt=attempt,
            )

            raw_text = self.run_generate(prompt)

            try:
                payload = extract_json_object(raw_text)

                candidates = self.parse_candidate_specs(
                    payload=payload,
                    incoming_message_count=incoming_message_count,
                )

                if len(candidates) == self.candidate_count:
                    return candidates, raw_text

                last_error = (
                    f"MLX вернула {len(candidates)} кандидатов "
                    f"вместо {self.candidate_count}."
                )

            except Exception as error:
                last_error = str(error)

        raise RuntimeError(f"MLX generation failed: {last_error}")

    def generate_candidates(
        self,
        history_messages: list[dict[str, Any]],
        incoming_messages: list[dict[str, Any]],
    ) -> tuple[list[list[str]], str]:
        candidate_specs, raw_response = self.generate_candidate_specs(
            history_messages=history_messages,
            incoming_messages=incoming_messages,
        )

        return [
            list(candidate["messages"])
            for candidate in candidate_specs
        ], raw_response
