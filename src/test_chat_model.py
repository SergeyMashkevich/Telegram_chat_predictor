from __future__ import annotations

import os
import subprocess
import sys
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
            "No chat selected. Run make run or make select-chat first."
        )

    return ADAPTERS_DIR / "chats" / str(active_chat["chat_id"]) / "lora"


def build_prompt(incoming_lines: list[str], model: str) -> str:
    incoming_text = "\n".join(incoming_lines)

    system_prompt = (
        "Imitate the user's style in a private Telegram chat. "
        "Respond as the user would most likely respond. "
        "Use the language naturally implied by the conversation. "
        "Do not explain the response."
    )

    user_prompt = (
        "Current incoming batch to answer:\n\n"
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
            "No chat selected. Run make run or make select-chat first."
        )

    model = resolve_active_mlx_model()

    adapter_path = adapter_dir()

    if not (adapter_path / "adapters.safetensors").exists():
        raise RuntimeError(
            f"Adapter not found: {adapter_path / 'adapters.safetensors'}"
        )

    print()
    print("MLX trained chat model test")
    print("===========================")
    print(f"chat:    {active_chat['title']}")
    print(f"model:   {model}")
    print(f"adapter: {adapter_path}")
    print()
    print("Enter an incoming batch from the chat partner.")
    print("An empty line finishes the input.")
    print()

    incoming_lines: list[str] = []

    while True:
        line = input(f"[{len(incoming_lines) + 1}] ").strip()

        if not line:
            break

        incoming_lines.append(line)

    if not incoming_lines:
        raise RuntimeError("Enter at least one message.")

    prompt = build_prompt(
        incoming_lines=incoming_lines,
        model=model,
    )

    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "generate",
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
