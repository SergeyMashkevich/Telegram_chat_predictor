from __future__ import annotations

from getpass import getpass
from typing import Any

from src.telegram_client import TdlibClient


def print_error(event: dict[str, Any]) -> None:
    code = event.get("code", "unknown")
    message = event.get("message", "unknown error")
    extra = event.get("@extra")

    print(f"Ошибка TDLib: code={code}, message={message}, extra={extra}")


def main() -> None:
    client = TdlibClient()

    client.send(
        {
            "@type": "getAuthorizationState",
            "@extra": "initial_authorization_state",
        }
    )

    print("TDLib запущена. Ожидание состояния авторизации...")

    while True:
        event = client.receive(timeout=1.0)

        if event is None:
            continue

        event_type = event.get("@type")

        if event_type == "error":
            print_error(event)
            continue

        if event_type == "user" and event.get("@extra") == "get_me":
            first_name = event.get("first_name", "")
            last_name = event.get("last_name", "")
            phone_number = event.get("phone_number", "")

            full_name = f"{first_name} {last_name}".strip()

            print(f"Авторизованный аккаунт: {full_name}")
            print(f"Номер телефона: +{phone_number}")

            client.send(
                {
                    "@type": "close",
                    "@extra": "close_after_authorization",
                }
            )
            continue

        state: dict[str, Any] | None = None

        if event_type == "updateAuthorizationState":
            state = event["authorization_state"]

        elif (
            isinstance(event_type, str)
            and event_type.startswith("authorizationState")
        ):
            state = event

        if state is None:
            continue

        state_type = state["@type"]

        print(f"Состояние: {state_type}")

        if state_type == "authorizationStateWaitTdlibParameters":
            request = client.build_tdlib_parameters()
            request["@extra"] = "set_tdlib_parameters"
            client.send(request)

        elif state_type == "authorizationStateWaitPhoneNumber":
            phone_number = input(
                "Введите номер телефона в международном формате, например +45...: "
            ).strip()

            client.send(
                {
                    "@type": "setAuthenticationPhoneNumber",
                    "phone_number": phone_number,
                    "settings": None,
                    "@extra": "set_phone_number",
                }
            )

        elif state_type == "authorizationStateWaitCode":
            code = input("Введите код подтверждения Telegram: ").strip()

            client.send(
                {
                    "@type": "checkAuthenticationCode",
                    "code": code,
                    "@extra": "check_authentication_code",
                }
            )

        elif state_type == "authorizationStateWaitPassword":
            password = getpass(
                "Введите пароль двухэтапной аутентификации: "
            )

            client.send(
                {
                    "@type": "checkAuthenticationPassword",
                    "password": password,
                    "@extra": "check_authentication_password",
                }
            )

        elif state_type == "authorizationStateWaitOtherDeviceConfirmation":
            link = state.get("link", "")
            print("Подтвердите вход на другом устройстве:")
            print(link)

        elif state_type == "authorizationStateReady":
            print("Авторизация завершена успешно.")

            client.send(
                {
                    "@type": "getMe",
                    "@extra": "get_me",
                }
            )

        elif state_type == "authorizationStateClosing":
            print("TDLib закрывает соединение...")

        elif state_type == "authorizationStateClosed":
            print("TDLib закрыта. Сессия сохранена в tdlib_data/.")
            break

        else:
            raise RuntimeError(
                f"Пока не поддерживается состояние авторизации: {state_type}"
            )


if __name__ == "__main__":
    main()