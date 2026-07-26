from __future__ import annotations

import os
import threading
import traceback

from dotenv import load_dotenv

from src.embedding_similarity import OllamaEmbeddingSimilarity
from src.response_matching_storage import (
    claim_next_response_for_matching,
    get_candidates_for_response_batch,
    initialize_response_matching_storage,
    mark_response_matching_failed,
    save_response_match,
)
from src.telegram_client import ENV_PATH


def join_messages(messages: list[str]) -> str:
    return "\n".join(
        str(message).strip()
        for message in messages
        if str(message).strip()
    )


class ResponseMatcherWorker:
    def __init__(self) -> None:
        load_dotenv(ENV_PATH)

        self.inferred_threshold = float(
            os.getenv("MATCH_INFERRED_THRESHOLD", "0.88")
        )
        self.independent_threshold = float(
            os.getenv("MATCH_INDEPENDENT_THRESHOLD", "0.55")
        )

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        initialize_response_matching_storage()

        self._thread = threading.Thread(
            target=self._run,
            name="response-matcher",
            daemon=True,
        )

        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=30.0)

    def _run(self) -> None:
        try:
            similarity = OllamaEmbeddingSimilarity()

        except Exception as error:
            print(
                f"[response matcher] Could not start: {error}",
                flush=True,
            )
            return

        while not self._stop_event.is_set():
            response = claim_next_response_for_matching()

            if response is None:
                self._stop_event.wait(1.0)
                continue

            self._process_response(
                similarity=similarity,
                response=response,
            )

    def _process_response(
        self,
        similarity: OllamaEmbeddingSimilarity,
        response: dict,
    ) -> None:
        response_id = int(response["response_id"])
        batch_id = int(response["batch_id"])
        final_text = join_messages(response["final_messages"])

        candidates = get_candidates_for_response_batch(batch_id)

        if not candidates or not final_text:
            save_response_match(
                response_id=response_id,
                candidate_id=None,
                similarity_score=None,
                attribution="unknown",
            )
            return

        try:
            texts = [final_text] + [
                join_messages(candidate["messages"])
                for candidate in candidates
            ]

            vectors = similarity.embed_texts(texts)
            final_vector = vectors[0]
            candidate_vectors = vectors[1:]

            best_candidate = None
            best_score = -1.0

            from src.embedding_similarity import cosine_similarity

            for candidate, vector in zip(candidates, candidate_vectors):
                score = cosine_similarity(final_vector, vector)

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate is None:
                save_response_match(
                    response_id=response_id,
                    candidate_id=None,
                    similarity_score=None,
                    attribution="unknown",
                )
                return

            if best_score >= self.inferred_threshold:
                attribution = "inferred"
            elif best_score < self.independent_threshold:
                attribution = "independent_response"
            else:
                attribution = "unknown"

            save_response_match(
                response_id=response_id,
                candidate_id=int(best_candidate["candidate_id"]),
                similarity_score=float(best_score),
                attribution=attribution,
            )

            print(
                f"[response matching] "
                f"response #{response_id}, batch #{batch_id}: "
                f"best candidate [{best_candidate['position']}], "
                f"similarity={best_score:.3f}, "
                f"attribution={attribution}",
                flush=True,
            )

        except Exception as error:
            mark_response_matching_failed(
                response_id=response_id,
                error_text=str(error),
            )

            print(
                f"[response matching error #{response_id}] {error}",
                flush=True,
            )

            traceback.print_exc()
