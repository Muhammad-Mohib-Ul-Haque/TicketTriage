"""Classifier hierarchy, evaluation and model selection (owned by Hassan Ul Haq).

OOP concepts demonstrated here:
    * Abstraction   — `TicketClassifier` is an ABC with abstract fit()/predict().
    * Encapsulation — model and vectorizer live in protected attributes
                      (`_model`, `_vectorizer`) behind a small public API.
    * Inheritance   — two concrete subclasses share the TF-IDF plumbing.
    * Overriding    — each subclass overrides fit()/predict() with its own estimator.
    * Polymorphism  — `evaluate_classifier()` / `compare_models()` accept *any*
                      TicketClassifier and call the same methods on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from . import config
from .exceptions import ModelNotTrainedError
from .text_utils import normalize_text, tokenize


def build_vectorizer() -> TfidfVectorizer:
    """Create a TF-IDF vectorizer wired to our own normaliser and tokenizer.

    Unigrams + bigrams capture short phrases such as "not received" or
    "reset password" that single words miss. sublinear_tf dampens repeated words.
    """
    return TfidfVectorizer(
        preprocessor=normalize_text,
        tokenizer=tokenize,
        token_pattern=None,  # silence warning: we supply our own tokenizer
        lowercase=False,  # normalize_text already lower-cases
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )


class TicketClassifier(ABC):
    """Abstract base for ticket classifiers: TF-IDF features + a scikit-learn estimator."""

    #: Human-readable name used in reports and charts. Subclasses override it.
    name: str = "Abstract classifier"

    def __init__(self, random_state: int = config.RANDOM_SEED) -> None:
        self._random_state = random_state
        self._vectorizer: TfidfVectorizer = build_vectorizer()
        self._model = None  # set by subclasses in fit()
        self._is_fitted = False

    # -- abstract interface -------------------------------------------------
    @abstractmethod
    def fit(self, X: Sequence[str], y: Sequence[str]) -> "TicketClassifier":
        """Fit vectorizer and model on training messages X and labels y."""

    @abstractmethod
    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Return predicted category labels for messages X."""

    # -- shared helpers -----------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """True once fit() has completed."""
        return self._is_fitted

    @property
    def classes_(self) -> np.ndarray:
        """Category labels in the column order used by predict_proba()."""
        self._require_fitted()
        return self._model.classes_

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._model is None:
            raise ModelNotTrainedError(f"{self.name} has not been trained yet.")

    def _fit_vectorizer(self, X: Sequence[str]):
        # The vectorizer is fitted on training data ONLY; test data is merely
        # transformed. Fitting on all data would leak test vocabulary/IDF.
        return self._vectorizer.fit_transform(X)

    def _transform(self, X: Sequence[str]):
        self._require_fitted()
        return self._vectorizer.transform(X)

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Class probabilities, shape (n_messages, n_classes)."""
        features = self._transform(X)  # checks fitted state before touching _model
        return self._model.predict_proba(features)

    def known_token_counts(self, X: Sequence[str]) -> np.ndarray:
        """How many n-gram features of each message exist in the training vocabulary.

        Zero means the model has never seen any of the words — the prediction
        is then essentially the class prior and must go to a human.
        """
        matrix = self._transform(X)
        return np.asarray((matrix > 0).sum(axis=1)).ravel()


class NaiveBayesClassifier(TicketClassifier):
    """Multinomial Naive Bayes over TF-IDF features — a fast, strong text baseline."""

    name = "Multinomial Naive Bayes"

    def __init__(self, alpha: float = 0.3, random_state: int = config.RANDOM_SEED) -> None:
        super().__init__(random_state)
        self._alpha = alpha

    def fit(self, X: Sequence[str], y: Sequence[str]) -> "NaiveBayesClassifier":
        features = self._fit_vectorizer(X)
        # MultinomialNB is deterministic, so random_state is not needed here.
        self._model = MultinomialNB(alpha=self._alpha)
        self._model.fit(features, y)
        self._is_fitted = True
        return self

    def predict(self, X: Sequence[str]) -> np.ndarray:
        features = self._transform(X)  # raises ModelNotTrainedError if unfitted
        return self._model.predict(features)


class LogisticTicketClassifier(TicketClassifier):
    """Multinomial logistic regression over TF-IDF features."""

    name = "Logistic Regression"

    def __init__(self, C: float = 5.0, random_state: int = config.RANDOM_SEED) -> None:
        super().__init__(random_state)
        self._C = C

    def fit(self, X: Sequence[str], y: Sequence[str]) -> "LogisticTicketClassifier":
        features = self._fit_vectorizer(X)
        # max_iter=2000 so lbfgs converges on ~20k sparse rows without warnings.
        self._model = LogisticRegression(
            C=self._C, max_iter=2000, random_state=self._random_state
        )
        self._model.fit(features, y)
        self._is_fitted = True
        return self

    def predict(self, X: Sequence[str]) -> np.ndarray:
        features = self._transform(X)  # raises ModelNotTrainedError if unfitted
        return self._model.predict(features)


# --- Evaluation & selection ---------------------------------------------------
@dataclass
class EvaluationResult:
    """Held-out metrics for one trained classifier."""

    name: str
    accuracy: float
    macro_f1: float
    report: str = field(repr=False)


@dataclass
class ModelComparison:
    """Outcome of training/evaluating several classifiers on the same split."""

    results: list[EvaluationResult]
    best: TicketClassifier
    best_result: EvaluationResult
    n_train: int
    n_test: int
    reason: str


def split_data(
    texts: Sequence[str], labels: Sequence[str], test_size: float = config.TEST_SIZE
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Stratified, seeded train/test split so every class keeps its proportion."""
    X_train, X_test, y_train, y_test = train_test_split(
        list(texts),
        list(labels),
        test_size=test_size,
        stratify=list(labels),
        random_state=config.RANDOM_SEED,
    )
    return X_train, X_test, y_train, y_test


def evaluate_classifier(
    clf: TicketClassifier,
    X_train: Sequence[str],
    y_train: Sequence[str],
    X_test: Sequence[str],
    y_test: Sequence[str],
) -> EvaluationResult:
    """Train and score ANY TicketClassifier — the polymorphic entry point."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return EvaluationResult(
        name=clf.name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        report=classification_report(y_test, y_pred, digits=3, zero_division=0),
    )


def compare_models(
    texts: Sequence[str],
    labels: Sequence[str],
    classifiers: Sequence[TicketClassifier] | None = None,
) -> ModelComparison:
    """Evaluate every classifier on the same held-out set and pick the best.

    Selection rule: highest macro-F1 (treats all five categories equally, so a
    model cannot win by only doing well on the largest class). Ties are broken
    by accuracy, then by list order.
    """
    if classifiers is None:
        classifiers = [NaiveBayesClassifier(), LogisticTicketClassifier()]
    X_train, X_test, y_train, y_test = split_data(texts, labels)

    results = [evaluate_classifier(c, X_train, y_train, X_test, y_test) for c in classifiers]
    # max() with a lambda key: macro-F1 first, accuracy as tie-breaker.
    best_idx = max(range(len(results)), key=lambda i: (results[i].macro_f1, results[i].accuracy))
    best_res = results[best_idx]
    others = [r for i, r in enumerate(results) if i != best_idx]
    margin = ", ".join(f"{r.name} {r.macro_f1:.3f}" for r in others)
    reason = (
        f"{best_res.name} has the highest held-out macro-F1 ({best_res.macro_f1:.3f} vs {margin}). "
        "Macro-F1 was used because it weights every category equally."
    )
    return ModelComparison(
        results=results,
        best=classifiers[best_idx],
        best_result=best_res,
        n_train=len(X_train),
        n_test=len(X_test),
        reason=reason,
    )
