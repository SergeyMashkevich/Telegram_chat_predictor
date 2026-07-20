from __future__ import annotations

from typing import Any


class FallbackPredictor:
    def __init__(
        self,
        primary: Any,
        fallback: Any | None,
        *,
        fallback_enabled: bool,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.fallback_enabled = fallback_enabled
        self.model = self._model_name()

    def _model_name(self) -> str:
        primary_model = getattr(self.primary, "model", "primary")

        if self.fallback is None:
            return str(primary_model)

        fallback_model = getattr(self.fallback, "model", "fallback")
        return f"{primary_model} | fallback={fallback_model}"

    def refresh_model(self) -> str:
        if hasattr(self.primary, "refresh_model"):
            self.primary.refresh_model()

        if self.fallback is not None and hasattr(self.fallback, "refresh_model"):
            self.fallback.refresh_model()

        self.model = self._model_name()
        return self.model

    def generate_candidate_specs(
        self,
        history_messages: list[dict[str, Any]],
        incoming_messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        try:
            return self.primary.generate_candidate_specs(
                history_messages=history_messages,
                incoming_messages=incoming_messages,
            )

        except Exception as primary_error:
            if not self.fallback_enabled or self.fallback is None:
                raise

            print(
                f"[generation fallback] Primary provider failed: {primary_error}",
                flush=True,
            )
            print(
                "[generation fallback] Trying fallback provider...",
                flush=True,
            )

            candidates, raw_response = self.fallback.generate_candidate_specs(
                history_messages=history_messages,
                incoming_messages=incoming_messages,
            )

            wrapped_raw_response = (
                "[fallback_used]\\n"
                f"primary_error: {primary_error}\\n\\n"
                f"{raw_response}"
            )

            return candidates, wrapped_raw_response

    def generate_candidates(
        self,
        history_messages: list[dict[str, Any]],
        incoming_messages: list[dict[str, Any]],
    ) -> tuple[list[list[str]], str]:
        specs, raw_response = self.generate_candidate_specs(
            history_messages=history_messages,
            incoming_messages=incoming_messages,
        )

        return [
            list(candidate["messages"])
            for candidate in specs
        ], raw_response
