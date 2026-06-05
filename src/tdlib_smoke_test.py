from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def load_tdlib() -> ctypes.CDLL:
    load_dotenv(ENV_PATH)

    configured_path = os.getenv("TDLIB_LIBRARY_PATH")
    if not configured_path:
        raise RuntimeError("TDLIB_LIBRARY_PATH не указан в .env")

    library_path = Path(configured_path)
    if not library_path.is_absolute():
        library_path = PROJECT_ROOT / library_path

    library_path = library_path.resolve()

    if not library_path.exists():
        raise FileNotFoundError(
            f"Библиотека TDLib не найдена: {library_path}"
        )

    try:
        tdjson = ctypes.CDLL(str(library_path))
    except OSError as error:
        raise RuntimeError(
            f"Не удалось загрузить TDLib: {error}"
        ) from error

    tdjson.td_execute.argtypes = [ctypes.c_char_p]
    tdjson.td_execute.restype = ctypes.c_char_p

    return tdjson


def execute(tdjson: ctypes.CDLL, request: dict[str, Any]) -> dict[str, Any]:
    request_json = json.dumps(request, ensure_ascii=False).encode("utf-8")
    response_ptr = tdjson.td_execute(request_json)

    if response_ptr is None:
        raise RuntimeError("TDLib вернула пустой ответ")

    response_json = response_ptr.decode("utf-8")
    return json.loads(response_json)


def get_string_option(tdjson: ctypes.CDLL, name: str) -> str:
    response = execute(
        tdjson,
        {
            "@type": "getOption",
            "name": name,
        },
    )

    if response.get("@type") == "error":
        raise RuntimeError(
            f"Ошибка TDLib для getOption({name!r}): {response}"
        )

    if response.get("@type") != "optionValueString":
        raise RuntimeError(
            f"Неожиданный ответ TDLib для getOption({name!r}): {response}"
        )

    return str(response["value"])


def main() -> None:
    tdjson = load_tdlib()

    version = get_string_option(tdjson, "version")
    commit_hash = get_string_option(tdjson, "commit_hash")

    print("OK: библиотека TDLib загружена")
    print(f"TDLib version: {version}")
    print(f"Commit hash: {commit_hash}")


if __name__ == "__main__":
    main()