from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from src.chat_context import get_active_chat
from src.adapter_metadata import resolve_active_mlx_model
from src.telegram_client import ENV_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = PROJECT_ROOT / "adapters"


def adapter_dir() -> Path:
    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "Чат не выбран. Сначала выполните make run или make select-chat."
        )

    return ADAPTERS_DIR / "chats" / str(active_chat["chat_id"]) / "lora"


def build_prompt(incoming_lines: list[str], model: str) -> str:
    incoming_text = "\n".join(incoming_lines)

    system_prompt = (
        "Имитируй стиль пользователя в личной переписке Telegram. "
        "Ответь так, как пользователь вероятнее всего ответил бы сам. "
        "Не объясняй ответ."
    )

    user_prompt = (
        "Текущий входящий блок, на который нужно ответить:\n\n"
        f"{incoming_text}"
    )

    if "qwen" in model.lower():
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


def main() -> None:
    load_dotenv(ENV_PATH)

    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "Чат не выбран. Сначала выполните make run или make select-chat."
        )

    model = resolve_active_mlx_model()

    adapter_path = adapter_dir()

    if not (adapter_path / "adapters.safetensors").exists():
        raise RuntimeError(
            f"Adapter не найден: {adapter_path / 'adapters.safetensors'}"
        )

    print()
    print("MLX trained chat model test")
    print("===========================")
    print(f"chat:    {active_chat['title']}")
    print(f"model:   {model}")
    print(f"adapter: {adapter_path}")
    print()
    print("Введите входящий блок собеседника.")
    print("Пустая строка завершает ввод.")
    print()

    incoming_lines: list[str] = []

    while True:
        line = input(f"[{len(incoming_lines) + 1}] ").strip()

        if not line:
            break

        incoming_lines.append(line)

    if not incoming_lines:
        raise RuntimeError("Нужно ввести хотя бы одно сообщение.")

    prompt = build_prompt(
        incoming_lines=incoming_lines,
        model=model,
    )

    command = [
        "mlx_lm.generate",
        "--model",
        model,
        "--adapter-path",
        str(adapter_path),
        "--prompt",
        prompt,
        "--max-tokens",
        os.getenv("TEST_MAX_TOKENS", "120"),
        "--temp",
        os.getenv("TEST_TEMPERATURE", "0.55"),
    ]

    print()
    print("Generating...")
    print()

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
