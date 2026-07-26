from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.active_model_metadata import write_active_model_metadata
from src.chat_context import get_active_chat
from src.adapter_metadata import write_active_adapter_metadata
from src.model_registry import suggested_trained_model_name
from src.telegram_client import ENV_PATH


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"


def dataset_dir() -> Path:
    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "No chat selected. Run make run or make select-chat first."
        )

    return DATASETS_DIR / "chats" / str(active_chat["chat_id"])


def adapter_dir() -> Path:
    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "No chat selected. Run make run or make select-chat first."
        )

    return ADAPTERS_DIR / "chats" / str(active_chat["chat_id"]) / "lora"


def main() -> None:
    load_dotenv(ENV_PATH)

    active_chat = get_active_chat()

    if active_chat is None:
        raise RuntimeError(
            "No chat selected. Run make run or make select-chat first."
        )

    model = os.getenv(
        "TRAIN_BASE_MODEL",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
    ).strip()

    iters = os.getenv("TRAIN_ITERS", "100").strip()
    batch_size = os.getenv("TRAIN_BATCH_SIZE", "1").strip()
    num_layers = os.getenv("TRAIN_NUM_LAYERS", "8").strip()

    data_path = dataset_dir() / "mlx"
    adapter_path = adapter_dir()
    adapter_path.parent.mkdir(parents=True, exist_ok=True)

    if not (data_path / "train.jsonl").exists():
        raise RuntimeError(
            f"Missing {data_path / 'train.jsonl'}. "
            "Run make prepare-training first."
        )

    print()
    print("Starting MLX LoRA training")
    print("==========================")
    print(f"chat:          {active_chat['title']}")
    print(f"chat_id:       {active_chat['chat_id']}")
    print(f"base model:    {model}")
    print(f"adapter path:  {adapter_path}")
    print(f"target name:   {suggested_trained_model_name()}")
    print(f"data:          {data_path}")
    print(f"iters:         {iters}")
    print(f"batch size:    {batch_size}")
    print(f"num layers:    {num_layers}")
    print()

    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model,
        "--train",
        "--data",
        str(data_path),
        "--adapter-path",
        str(adapter_path),
        "--iters",
        iters,
        "--batch-size",
        batch_size,
        "--num-layers",
        num_layers,
        "--mask-prompt",
    ]

    subprocess.run(command, check=True)

    metadata_path = write_active_model_metadata(
        base_model=model,
        source="train_chat_model",
        extra={
            "iters": int(iters),
            "batch_size": int(batch_size),
            "num_layers": int(num_layers),
            "adapter_path": str(adapter_path),
        },
    )

    print(f"active metadata: {metadata_path}")

    metadata_path = write_active_adapter_metadata(
        base_model=model,
        source="train_chat_model",
        extra={
            "iters": int(iters),
            "batch_size": int(batch_size),
            "num_layers": int(num_layers),
            "data_path": str(data_path),
        },
    )

    print()
    print("Training finished")
    print("=================")
    print(f"adapter: {adapter_path}")
    print(f"metadata: {metadata_path}")
    print()
    print("Next, test the adapter with mlx_lm.generate,")
    print("then decide how to connect it to the main application.")


if __name__ == "__main__":
    main()
