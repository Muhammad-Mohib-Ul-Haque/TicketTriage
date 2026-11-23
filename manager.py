"""TicketManager: validation, routing, prediction records and history I/O.

Owned jointly: routing (Hassan Ul Haq), JSON/CSV I/O and history (Mohib Ul Haque).

The manager only RECOMMENDS a category and team. It never contacts a customer
or edits any real record — every result is a suggestion for a human agent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import config
from .data_loader import read_csv_safely
from .exceptions import DatasetError, HistoryError, InvalidMessageError
from .models import TicketClassifier
from .text_utils import normalize_text

BATCH_TEXT_COLUMNS: tuple[str, ...] = ("message", "instruction", "text")


@dataclass
class Prediction:
    """One classified ticket, ready to be shown or serialised."""

    timestamp: str
    message: str
    category: str
    confidence: float
    team: str
    needs_review: bool
    review_reason: str
    model: str


class TicketManager:
    """Validates messages, applies a classifier, routes to teams and keeps history."""

    def __init__(
        self,
        classifier: TicketClassifier,
        history_path: Path = config.HISTORY_PATH,
        threshold: float = config.CONFIDENCE_THRESHOLD,
    ) -> None:
        self._classifier = classifier
        self._history_path = history_path
        self._threshold = threshold

    # -- validation ---------------------------------------------------------
    @staticmethod
    def validate_message(message: object) -> str:
        """Return the stripped message or raise InvalidMessageError explaining why."""
        if not isinstance(message, str):
            raise InvalidMessageError("Message must be text.")
        text = message.strip()
        if not text:
            raise InvalidMessageError("Message is empty.")
        if len(text) < config.MIN_MESSAGE_CHARS:
            raise InvalidMessageError(
                f"Message is too short (minimum {config.MIN_MESSAGE_CHARS} characters)."
            )
        if len(text) > config.MAX_MESSAGE_CHARS:
            raise InvalidMessageError(
                f"Message is too long (maximum {config.MAX_MESSAGE_CHARS} characters)."
            )
        if not normalize_text(text):
            raise InvalidMessageError("Message contains no letters to analyse.")
        return text

    @staticmethod
    def route(category: str) -> str:
        """Map a category to its responsible support team."""
        return config.TEAM_MAP.get(category, config.HUMAN_REVIEW_TEAM)

    # -- classification -----------------------------------------------------
    def classify_many(self, messages: Sequence[str]) -> list[Prediction]:
        """Classify already-validated messages in one vectorised pass."""
        if not messages:
            return []
        probs = self._classifier.predict_proba(messages)
        classes = self._classifier.classes_
        best_idx = probs.argmax(axis=1)
        best_prob = probs[np.arange(len(messages)), best_idx]
        known = self._classifier.known_token_counts(messages)
        stamp = datetime.now().isoformat(timespec="seconds")

        out: list[Prediction] = []
        for msg, idx, prob, n_known in zip(messages, best_idx, best_prob, known):
            category = str(classes[idx])
            if n_known == 0:
                reason = "No known vocabulary — message may be off-topic or unsupported."
            elif prob < self._threshold:
                reason = f"Low confidence ({prob:.0%} < {self._threshold:.0%})."
            else:
                reason = ""
            needs_review = bool(reason)
            out.append(
                Prediction(
                    timestamp=stamp,
                    message=msg,
                    category=category,
                    confidence=round(float(prob), 4),
                    # Suggest the team even when flagged, but mark it for a human first.
                    team=config.HUMAN_REVIEW_TEAM if needs_review else self.route(category),
                    needs_review=needs_review,
                    review_reason=reason,
                    model=self._classifier.name,
                )
            )
        return out

    def classify(self, message: str) -> Prediction:
        """Validate and classify a single message."""
        return self.classify_many([self.validate_message(message)])[0]

    def classify_csv(self, csv_path: Path) -> tuple[list[Prediction], list[tuple[int, str]]]:
        """Classify every row of a CSV.

        The CSV must have one of the columns: message, instruction or text.

        Returns:
            (predictions for valid rows, list of (row_number, error) for skipped rows)
        """
        df = read_csv_safely(csv_path)
        col = next((c for c in BATCH_TEXT_COLUMNS if c in df.columns), None)
        if col is None:
            raise DatasetError(
                f"CSV needs a text column named one of: {', '.join(BATCH_TEXT_COLUMNS)}. "
                f"Found: {', '.join(map(str, df.columns))}"
            )
        valid: list[str] = []
        skipped: list[tuple[int, str]] = []
        # Row numbers are 1-based + header, matching what a spreadsheet shows.
        for row_no, raw in enumerate(df[col].tolist(), start=2):
            try:
                valid.append(self.validate_message(raw if isinstance(raw, str) else ""))
            except InvalidMessageError as exc:
                skipped.append((row_no, str(exc)))
        return self.classify_many(valid), skipped

    # -- persistence --------------------------------------------------------
    def load_history(self) -> list[dict]:
        """Return saved predictions (empty list if no history yet)."""
        if not self._history_path.exists():
            return []
        try:
            data = json.loads(self._history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HistoryError(
                f"History file is corrupted ({self._history_path.name}): {exc}. "
                "Rename or delete it to start a fresh history."
            ) from exc
        except OSError as exc:
            raise HistoryError(f"Could not read history: {exc}") from exc
        if not isinstance(data, list):
            raise HistoryError("History file has an unexpected format (expected a JSON list).")
        return data

    def save_to_history(self, predictions: Sequence[Prediction]) -> int:
        """Append predictions to the JSON history file; returns new total count."""
        history = self.load_history()
        history.extend(asdict(p) for p in predictions)
        self._write_json(self._history_path, history)
        return len(history)

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file then replace, so a crash mid-write cannot
            # leave a half-written (corrupt) history file.
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            raise HistoryError(f"Could not write {path.name}: {exc}") from exc

    @staticmethod
    def export_results(predictions: Sequence[Prediction], stem: str) -> tuple[Path, Path]:
        """Save predictions as both CSV and JSON in outputs/; returns the two paths."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = config.OUTPUT_DIR / f"{stem}_{stamp}.csv"
        json_path = csv_path.with_suffix(".json")
        rows = [asdict(p) for p in predictions]
        try:
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
        except OSError as exc:
            raise HistoryError(f"Could not write {csv_path.name}: {exc}") from exc
        TicketManager._write_json(json_path, rows)
        return csv_path, json_path

    @staticmethod
    def summarise(predictions: Sequence[Prediction]) -> pd.DataFrame:
        """Per-category count, mean confidence and review count for a batch."""
        if not predictions:
            return pd.DataFrame(columns=["count", "mean_confidence", "needs_review"])
        df = pd.DataFrame([asdict(p) for p in predictions])
        return (
            df.groupby("category")
            .agg(
                count=("message", "size"),
                mean_confidence=("confidence", "mean"),
                needs_review=("needs_review", "sum"),
            )
            .round({"mean_confidence": 3})
            .sort_values("count", ascending=False)
        )
