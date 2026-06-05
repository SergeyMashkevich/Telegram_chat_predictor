from __future__ import annotations

import os
import threading
import traceback

from dotenv import load_dotenv

from src.prediction_storage import (
    claim_next_ready_batch,
    get_batch_info,
    get_recent_messages_before,
    initialize_prediction_storage,
    mark_generation_failed,
    recover_stuck_generating_batches,
    save_candidates_if_generating,
    update_candidate_scores,
)
from src.predictor import OllamaPredictor
from src.ranking import CandidateRanker
from src.storage import get_batch_messages
from src.telegram_client import ENV_PATH


def _preview(text: str, limit: int = 70) -> str:
    compact = " ".join(text.split())

    if len(compact) <= limit:
        return compact

    return f"{compact[:limit - 3]}..."


class PredictionWorker:
    def __init__(self, chat_id: int) -> None:
        load_dotenv(ENV_PATH)

        self.chat_id = int(chat_id)
        self.context_limit = int(
            os.getenv("RECENT_CONTEXT_MAX_MESSAGES", "100")
        )

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        initialize_prediction_storage()
        recover_stuck_generating_batches(self.chat_id)

        self._thread = threading.Thread(
            target=self._run,
            name="prediction-worker",
            daemon=True,
        )

        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=30.0)

    def _run(self) -> None:
        try:
            predictor = OllamaPredictor()

        except Exception as error:
            print(
                f"[prediction worker] Не удалось запустить predictor: {error}",
                flush=True,
            )
            return

        try:
            ranker: CandidateRanker | None = CandidateRanker()
            print("[prediction worker] Ranking включён.", flush=True)

        except Exception as error:
            ranker = None
            print(
                f"[prediction worker] Ranking отключён: {error}",
                flush=True,
            )

        while not self._stop_event.is_set():
            batch_id = claim_next_ready_batch(
                chat_id=self.chat_id,
                model=predictor.model,
            )

            if batch_id is None:
                self._stop_event.wait(0.5)
                continue

            self._process_batch(
                predictor=predictor,
                ranker=ranker,
                batch_id=batch_id,
            )

    def _process_batch(
        self,
        predictor: OllamaPredictor,
        ranker: CandidateRanker | None,
        batch_id: int,
    ) -> None:
        batch_info = get_batch_info(batch_id)

        if batch_info is None:
            mark_generation_failed(
                batch_id,
                "Входящий блок не найден.",
            )
            return

        incoming_messages = get_batch_messages(batch_id)

        if not incoming_messages:
            mark_generation_failed(
                batch_id,
                "Входящий блок не содержит сообщений.",
            )

            print(
                f"[ошибка генерации для блока #{batch_id}] "
                "блок не содержит сообщений.",
                flush=True,
            )
            return

        history_messages = get_recent_messages_before(
            chat_id=self.chat_id,
            before_message_id=int(batch_info["first_message_id"]),
            limit=self.context_limit,
        )

        print(
            f"[генерация кандидатов для блока #{batch_id}]",
            flush=True,
        )

        try:
            candidate_specs, raw_response = predictor.generate_candidate_specs(
                history_messages=history_messages,
                incoming_messages=incoming_messages,
            )

            if ranker is not None:
                candidate_specs = ranker.rank_candidates(
                    candidates=candidate_specs,
                    history_messages=history_messages,
                    incoming_messages=incoming_messages,
                )

        except Exception as error:
            error_text = str(error)

            mark_generation_failed(
                batch_id=batch_id,
                error_text=error_text,
            )

            print(
                f"[ошибка генерации для блока #{batch_id}] "
                f"{error_text}",
                flush=True,
            )

            traceback.print_exc()
            return

        saved = save_candidates_if_generating(
            batch_id=batch_id,
            candidates=candidate_specs,
            raw_response=raw_response,
        )

        if not saved:
            print(
                f"[кандидаты для блока #{batch_id}] "
                "не сохранены: блок уже был отвечен вручную.",
                flush=True,
            )
            return

        try:
            update_candidate_scores(
                batch_id=batch_id,
                ranked_candidates=candidate_specs,
            )
        except Exception as error:
            print(
                f"[ranking] Не удалось сохранить score для блока #{batch_id}: {error}",
                flush=True,
            )

        print()
        print(f"[кандидаты для блока #{batch_id}]")

        for position, candidate in enumerate(
            candidate_specs,
            start=1,
        ):
            reply_index = int(
                candidate.get("reply_to_incoming_index", 0)
            )

            score = candidate.get("final_score")

            print()
            if score is None:
                print(f"[{position}]")
            else:
                print(f"[{position}] score={score:.3f}")

            if reply_index == 0:
                print("(без reply)")
            elif 1 <= reply_index <= len(incoming_messages):
                target_text = str(
                    incoming_messages[reply_index - 1].get("text", "")
                )
                print(
                    f'(reply на [{reply_index}] "{_preview(target_text)}")'
                )
            else:
                print("(reply указан некорректно, будет проигнорирован)")

            for message in candidate["messages"]:
                print(message)

        print()
        print(
            f"Для отправки: send {batch_id} <номер>",
            flush=True,
        )
        print()
