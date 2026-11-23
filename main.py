"""TicketTriage console entry point.

Run:  python main.py

Console integration by Mohib Ul Haque; routing/model wiring by Hassan Ul Haq.
The app only RECOMMENDS a category and team; a human agent makes the final call.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from tickettriage import analytics, config
from tickettriage.api_client import ProductAPIClient
from tickettriage.data_loader import LoadedDataset, load_dataset
from tickettriage.exceptions import TicketTriageError
from tickettriage.manager import Prediction, TicketManager
from tickettriage.models import ModelComparison, compare_models

LINE = "-" * 64


def ask(prompt: str) -> str:
    """input() wrapper; EOF (Ctrl+D / closed stdin) is treated as a request to quit."""
    try:
        return input(prompt)
    except EOFError:
        raise KeyboardInterrupt from None


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    """Re-prompt until the user answers y/n (Enter accepts the default)."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = ask(f"{prompt} {hint}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("  Please answer y or n.")


def ask_int(prompt: str, minimum: int, maximum: int, default: int | None = None) -> int:
    """Re-prompt until the user enters an integer within [minimum, maximum]."""
    while True:
        raw = ask(prompt).strip()
        if not raw and default is not None:
            return default
        if raw.isdigit() and minimum <= int(raw) <= maximum:
            return int(raw)
        print(f"  Please enter a whole number from {minimum} to {maximum}.")


def print_prediction(p: Prediction) -> None:
    """Show one prediction in a compact, readable block."""
    print(f"  Category    : {p.category}")
    print(f"  Confidence  : {p.confidence:.1%}")
    print(f"  Route to    : {p.team}")
    if p.needs_review:
        print(f"  ⚠ Review    : {p.review_reason}")
        print(f"  Suggested   : {TicketManager.route(p.category)} (after human check)")


class TriageApp:
    """Menu-driven console application holding the session state."""

    def __init__(self) -> None:
        self.dataset: LoadedDataset | None = None
        self.comparison: ModelComparison | None = None
        self.manager: TicketManager | None = None
        self.api = ProductAPIClient()

    # -- startup ------------------------------------------------------------
    def train(self) -> None:
        """Load data, train both classifiers and keep the better one."""
        print("Loading dataset...")
        self.dataset = load_dataset()
        ds = self.dataset
        label = "SYNTHETIC SAMPLE (demo only)" if ds.is_sample else "Bitext full dataset"
        print(f"  Source : {ds.source_path.name}  [{label}]")
        print(
            f"  Rows   : {ds.rows_raw:,} raw -> {len(ds.frame):,} usable "
            f"(dropped {ds.rows_missing_dropped} missing/empty, "
            f"{ds.rows_unmapped_dropped} out-of-scope categories, "
            f"{ds.rows_duplicate_dropped} duplicates)"
        )
        print("Training Naive Bayes and Logistic Regression...")
        df = ds.frame
        self.comparison = compare_models(df["instruction"], df["category"])
        self.manager = TicketManager(self.comparison.best)
        best = self.comparison.best_result
        print(
            f"  Selected: {best.name} (accuracy {best.accuracy:.3f}, macro-F1 {best.macro_f1:.3f})"
        )

    # -- menu actions -------------------------------------------------------
    def single_ticket(self) -> None:
        """Classify one typed message and optionally save it."""
        message = ask("Enter the customer message (blank to cancel):\n> ")
        if not message.strip():
            print("  Cancelled.")
            return
        prediction = self.manager.classify(message)
        print(LINE)
        print_prediction(prediction)
        if ask_yes_no("Save this prediction to history?", default=True):
            total = self.manager.save_to_history([prediction])
            print(f"  Saved. History now holds {total} prediction(s).")

    def batch_mode(self) -> None:
        """Classify every row of a user-supplied CSV and export the results."""
        default = config.SAMPLE_BATCH_PATH
        raw = ask(f"CSV path (Enter for {default.relative_to(config.PROJECT_ROOT)}): ").strip()
        # Accept paths pasted with quotes (common when copying from Windows Explorer).
        path = Path(raw.strip('"').strip("'")).expanduser() if raw else default
        predictions, skipped = self.manager.classify_csv(path)
        print(LINE)
        print(f"  Classified {len(predictions)} row(s); skipped {len(skipped)} invalid row(s).")
        for row_no, reason in skipped[:10]:
            print(f"    row {row_no}: {reason}")
        if not predictions:
            return
        print("\n  Summary by category:")
        print(TicketManager.summarise(predictions).to_string().replace("\n", "\n  ").join(["  ", ""]))
        flagged = sum(p.needs_review for p in predictions)
        print(f"\n  Flagged for human review: {flagged}")
        csv_path, json_path = TicketManager.export_results(predictions, "batch_results")
        print(f"  Saved: {csv_path.relative_to(config.PROJECT_ROOT)}")
        print(f"         {json_path.relative_to(config.PROJECT_ROOT)}")
        if ask_yes_no("Also append these to history?"):
            total = self.manager.save_to_history(predictions)
            print(f"  History now holds {total} prediction(s).")

    def history_mode(self) -> None:
        """List the most recent saved predictions."""
        history = self.manager.load_history()
        if not history:
            print("  No saved predictions yet.")
            return
        n = ask_int(f"How many recent entries to show? (1-{len(history)}, Enter=10): ",
                    1, len(history), default=min(10, len(history)))
        frame = pd.DataFrame(history[-n:])
        frame["message"] = frame["message"].str.slice(0, 45)
        frame["review"] = frame["needs_review"].map(lambda flag: "YES" if flag else "")
        cols = ["timestamp", "message", "category", "confidence", "review"]
        print(frame[cols].to_string(index=False))
        print(f"\n  Total in history: {len(history)}; category counts:")
        print(pd.DataFrame(history)["category"].value_counts().to_string())

    def analytics_mode(self) -> None:
        """Run Pandas/NumPy summaries and save the three charts."""
        df = self.dataset.frame
        print("Ticket length statistics:")
        print(analytics.length_statistics(df).to_string())
        print("\nPer-category summary (groupby):")
        print(analytics.category_summary(df).to_string())
        print("\nTop (category, intent) pairs — exploratory only:")
        print(analytics.intent_summary(df, top=8).to_string(index=False))

        m = analytics.analyse_matrix(df)
        print("\nNumPy feature matrix (rows = tickets, cols = features):")
        print("  Features      :", ", ".join(m.feature_names))
        print("  Column means  :", m.column_means)
        print("  Column std    :", m.column_std)
        print("  Mean features per category (boolean-mask slices):")
        print(m.per_category_means.to_string().replace("\n", "\n    ").join(["    ", ""]))
        print(f"  Long tickets (> {m.long_ticket_threshold:.0f} words, top 10%): {m.long_ticket_count}")
        if not m.long_ticket_share_by_category.empty:
            shares = ", ".join(f"{k} {v}%" for k, v in m.long_ticket_share_by_category.items())
            print(f"    share by category: {shares}")
        print(f"  Length outliers (|z| > 3): {m.zscore_extremes}")

        paths = analytics.generate_all_charts(df, self.comparison.results)
        print("\nCharts saved:")
        for p in paths:
            print(f"  {p.relative_to(config.PROJECT_ROOT)}")

    def product_mode(self) -> None:
        """Look up product context from DummyJSON (cached, offline-safe)."""
        product_id = ask_int("Product ID (1-194, Enter=1): ", 1, 194, default=1)
        use_live = ask_yes_no("Try the live API? (No = use cache only)", default=True)
        result = self.api.get_product(product_id, use_live=use_live)
        notes = {
            "live": "fresh from dummyjson.com (cache updated)",
            "cache": "from local cache — API unreachable or skipped",
            "bundled sample": "bundled offline sample — may not match the live API",
        }
        print(LINE)
        print(f"  Source: {notes[result.source]}")
        print(ProductAPIClient.format_product(result.data))

    def comparison_mode(self) -> None:
        """Show held-out metrics for both models and why one was chosen."""
        c = self.comparison
        print(f"Stratified split: {c.n_train:,} train / {c.n_test:,} test (seed {config.RANDOM_SEED})")
        print(f"{'Model':<26}{'Accuracy':>10}{'Macro-F1':>10}")
        for r in c.results:
            marker = "  <- selected" if r is c.best_result else ""
            print(f"{r.name:<26}{r.accuracy:>10.3f}{r.macro_f1:>10.3f}{marker}")
        print(f"\nReason: {c.reason}")
        if ask_yes_no("Show per-category report for the selected model?"):
            print(c.best_result.report)
        if self.dataset.is_sample:
            print("Note: scores come from the small SYNTHETIC sample and are not "
                  "representative. Download the Bitext CSV for meaningful numbers.")

    # -- loop ---------------------------------------------------------------
    def run(self) -> None:
        """Main menu loop."""
        actions: dict[str, tuple[str, Callable[[], None]]] = {
            "1": ("Classify a single ticket", self.single_ticket),
            "2": ("Batch classify a CSV file", self.batch_mode),
            "3": ("Review prediction history", self.history_mode),
            "4": ("Analytics + charts", self.analytics_mode),
            "5": ("Product lookup (DummyJSON)", self.product_mode),
            "6": ("Model comparison", self.comparison_mode),
            "0": ("Exit", lambda: None),
        }
        while True:
            print(f"\n{LINE}\n TicketTriage — main menu\n{LINE}")
            for key, (label, _) in actions.items():
                print(f"  {key}. {label}")
            choice = ask("Choose an option: ").strip()
            if choice not in actions:
                print("  Invalid choice — enter one of: " + ", ".join(actions))
                continue
            if choice == "0":
                print("Goodbye.")
                return
            print(LINE)
            try:
                actions[choice][1]()
            except TicketTriageError as exc:
                print(f"  Error: {exc}")
            except ValueError as exc:
                print(f"  Invalid input: {exc}")
            except KeyboardInterrupt:
                print("\n  Cancelled — back to menu.")
            except Exception as exc:  # noqa: BLE001 — keep the session alive on a bug
                print(f"  Unexpected error: {type(exc).__name__}: {exc}")


def main() -> int:
    """Entry point; returns a process exit code."""
    print("TicketTriage — AI Customer Support Ticket Classifier")
    print("Recommendations only: a human agent reviews every ticket.\n")
    app = TriageApp()
    try:
        app.train()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted — goodbye.")
    except TicketTriageError as exc:
        print(f"Startup failed: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 — last-resort guard: no raw traceback for users
        print(f"Unexpected error: {type(exc).__name__}: {exc}")
        print("Please report this with the steps that caused it.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
