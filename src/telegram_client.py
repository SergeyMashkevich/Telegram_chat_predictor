from __future__ import annotations

import ctypes
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def require_env(name: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(f"Переменная {name} не указана в .env")

    return value.strip()


def resolve_project_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


class TdlibClient:
    def __init__(self) -> None:
        load_dotenv(ENV_PATH)

        library_path = resolve_project_path(require_env("TDLIB_LIBRARY_PATH"))

        if not library_path.exists():
            raise FileNotFoundError(
                f"Библиотека TDLib не найдена: {library_path}"
            )

        self._tdjson = ctypes.CDLL(str(library_path))
        self._buffered_events: deque[dict[str, Any]] = deque()

        self._tdjson.td_execute.argtypes = [ctypes.c_char_p]
        self._tdjson.td_execute.restype = ctypes.c_char_p

        self._tdjson.td_create_client_id.argtypes = []
        self._tdjson.td_create_client_id.restype = ctypes.c_int

        self._tdjson.td_send.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
        ]
        self._tdjson.td_send.restype = None

        self._tdjson.td_receive.argtypes = [
            ctypes.c_double,
        ]
        self._tdjson.td_receive.restype = ctypes.c_char_p

        self.execute(
            {
                "@type": "setLogVerbosityLevel",
                "new_verbosity_level": 1,
            }
        )

        self.client_id = self._tdjson.td_create_client_id()

    def execute(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_json = json.dumps(
            request,
            ensure_ascii=False,
        ).encode("utf-8")

        response_ptr = self._tdjson.td_execute(request_json)

        if response_ptr is None:
            return None

        return json.loads(response_ptr.decode("utf-8"))

    def send(self, request: dict[str, Any]) -> None:
        request_json = json.dumps(
            request,
            ensure_ascii=False,
        ).encode("utf-8")

        self._tdjson.td_send(
            self.client_id,
            request_json,
        )

    def _raw_receive(self, timeout: float) -> dict[str, Any] | None:
        response_ptr = self._tdjson.td_receive(timeout)

        if response_ptr is None:
            return None

        response = json.loads(response_ptr.decode("utf-8"))

        response_client_id = response.get("@client_id")

        if (
            response_client_id is not None
            and response_client_id != self.client_id
        ):
            return None

        return response

    def receive(self, timeout: float = 1.0) -> dict[str, Any] | None:
        if self._buffered_events:
            return self._buffered_events.popleft()

        return self._raw_receive(timeout)

    def request(
        self,
        request: dict[str, Any],
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        extra = request.get("@extra") or f"request:{uuid4().hex}"

        payload = dict(request)
        payload["@extra"] = extra

        self.send(payload)

        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                raise TimeoutError(
                    f"TDLib не ответила на запрос: {request.get('@type')}"
                )

            event = self._raw_receive(min(1.0, remaining))

            if event is None:
                continue

            if event.get("@extra") == extra:
                if event.get("@type") == "error":
                    raise RuntimeError(
                        f"Ошибка TDLib: {event.get('code')} "
                        f"{event.get('message')}"
                    )

                return event

            self._buffered_events.append(event)

    def build_tdlib_parameters(self) -> dict[str, Any]:
        database_directory = resolve_project_path(
            require_env("TDLIB_DATABASE_DIR")
        )
        files_directory = resolve_project_path(
            require_env("TDLIB_FILES_DIR")
        )

        database_directory.mkdir(parents=True, exist_ok=True)
        files_directory.mkdir(parents=True, exist_ok=True)

        return {
            "@type": "setTdlibParameters",
            "use_test_dc": False,
            "database_directory": str(database_directory),
            "files_directory": str(files_directory),
            "database_encryption_key": require_env(
                "TDLIB_DATABASE_ENCRYPTION_KEY"
            ),
            "use_file_database": True,
            "use_chat_info_database": True,
            "use_message_database": True,
            "use_secret_chats": False,
            "api_id": int(require_env("TELEGRAM_API_ID")),
            "api_hash": require_env("TELEGRAM_API_HASH"),
            "system_language_code": "ru",
            "device_model": "MacBook",
            "system_version": "macOS",
            "application_version": "0.1.0",
        }
