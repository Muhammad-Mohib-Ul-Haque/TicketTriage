"""Loading the dataset, Pandas/NumPy analysis and the three charts.

Data loading and analysis: Mohib Ul Haque. Charts: Hassan Ul Haq.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # draw charts straight into PNG files (no pop-up window needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exceptions import TicketTriageError
from text_utils import normalize_text

# Paths are built from this file's folder, so the project works wherever it is cloned.
PROJECT_DIR = Path(__file__).parent
FULL_DATASET = PROJECT_DIR / "data" / "bitext_customer_support.csv"
SAMPLE_DATASET = PROJECT_DIR / "data" / "sample_tickets.csv"
CHART_DIR = PROJECT_DIR / "outputs" / "charts"

# Bitext category  ->  our category.
# SHIPPING (changing the shipping address) goes to Delivery because the same team handles it.
# Bitext categories NOT listed here (CONTACT, INVOICE, FEEDBACK, ...) are removed.
CATEGORY_MAP = {
    "ACCOUNT": "Account",
    "ORDER": "Order",
    "DELIVERY": "Delivery",
    "SHIPPING": "Delivery",
    "PAYMENT": "Payment",
    "REFUND": "Refund",
}
CATEGORIES = ["Account", "Order", "Delivery", "Payment", "Refund"]


def read_csv_file(path) -> pd.DataFrame:
    """Read a CSV file, turning common problems into a friendly TicketTriageError."""
    path = Path(path)
    if not path.is_file():
        raise TicketTriageError(f"File not found: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise TicketTriageError(f"The CSV file is empty: {path.name}")
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise TicketTriageError(f"The CSV file is badly formatted ({path.name}): {error}")


def load_dataset():
    """Load the full Bitext CSV if it exists, otherwise the synthetic sample, and clean it.

    Returns (clean_dataframe, info) where info is a dictionary describing what happened.
    """
    if FULL_DATASET.is_file():
        path, is_sample = FULL_DATASET, False
    else:
        path, is_sample = SAMPLE_DATASET, True

    df = read_csv_file(path)
    for column in ["instruction", "category"]:
        if column not in df.columns:
            raise TicketTriageError(f"Dataset is missing the '{column}' column.")
    rows_raw = len(df)

    # 1) Missing values: drop rows with no message or no category.
    #    intent is only used for exploring the data, so we fill its gaps instead of dropping.
    df = df.dropna(subset=["instruction", "category"])
    if "intent" not in df.columns:
        df["intent"] = "unknown"
    df["intent"] = df["intent"].fillna("unknown")
    # Also drop messages that become empty after cleaning (e.g. "!!! ???").
    df = df[df["instruction"].apply(normalize_text) != ""]
    rows_missing = rows_raw - len(df)

    # 2) Filtering: keep only rows whose category is in our map, then rename them.
    before = len(df)
    df["category"] = df["category"].str.strip().str.upper()
    df = df[df["category"].isin(CATEGORY_MAP.keys())]
    df["category"] = df["category"].map(CATEGORY_MAP)
    rows_out_of_scope = before - len(df)

    # 3) Duplicates: the same message must not appear in both training and test data.
    before = len(df)
    df = df.drop_duplicates(subset=["instruction"])
    rows_duplicate = before - len(df)

    # 4) New columns used in the analysis.
    df["length_chars"] = df["instruction"].str.len()
    df["length_words"] = df["instruction"].str.split().str.len()

    # The stratified split needs at least 2 examples of every category.
    counts = df["category"].value_counts()
    for category in CATEGORIES:
        if counts.get(category, 0) < 2:
            raise TicketTriageError(f"Not enough rows for category '{category}'.")

    info = {
        "file": path.name,
        "is_sample": is_sample,
        "rows_raw": rows_raw,
        "rows_missing": rows_missing,
        "rows_out_of_scope": rows_out_of_scope,
        "rows_duplicate": rows_duplicate,
        "rows_clean": len(df),
    }
    return df.reset_index(drop=True), info


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pandas groupby: how many tickets each category has and how long they are."""
    summary = df.groupby("category").agg(
        tickets=("instruction", "count"),
        mean_words=("length_words", "mean"),
        max_words=("length_words", "max"),
        intents=("intent", "nunique"),
    )
    summary["share_%"] = summary["tickets"] / summary["tickets"].sum() * 100
    return summary.round(1)


def numpy_analysis(df: pd.DataFrame) -> dict:
    """NumPy on a 2-D matrix: rows = tickets, columns = [characters, words].

    Shows: building a 2-D array, axis aggregates, slicing, boolean masks and broadcasting.
    Returns a dictionary of results.
    """
    # Build the 2-D matrix (shape: number_of_tickets x 2).
    matrix = np.column_stack([df["length_chars"].to_numpy(), df["length_words"].to_numpy()])
    labels = df["category"].to_numpy()

    # Axis aggregates: axis=0 means "down the rows", giving one value per column.
    column_means = matrix.mean(axis=0)
    column_max = matrix.max(axis=0)

    # Slicing: matrix[:, 1] = every row, column 1 (the word counts).
    words = matrix[:, 1]

    # Broadcasting: subtract the 2 column means from EVERY row in one step.
    # (n x 2 matrix) - (2 values) works because NumPy repeats the 2 values for each row.
    centered = matrix - column_means
    above_average_words = int((centered[:, 1] > 0).sum())

    # Boolean masks: labels == "Refund" gives True/False per row; matrix[mask] keeps the True rows.
    mean_words_per_category = {}
    for category in CATEGORIES:
        mask = labels == category
        mean_words_per_category[category] = round(float(matrix[mask, 1].mean()), 2)

    long_mask = words > 12
    return {
        "shape": matrix.shape,
        "column_means": column_means.round(2),
        "column_max": column_max,
        "above_average_words": above_average_words,
        "mean_words_per_category": mean_words_per_category,
        "long_tickets": int(long_mask.sum()),
    }


def show_analysis(df: pd.DataFrame) -> None:
    """Print all Pandas and NumPy results to the console."""
    print("Ticket length statistics:")
    print(df[["length_chars", "length_words"]].describe().round(2))

    print("\nPer-category summary (groupby):")
    print(category_summary(df))

    print("\nMost common intents (for exploring only, not used by the model):")
    print(df["intent"].value_counts().head(8))

    result = numpy_analysis(df)
    print("\nNumPy matrix analysis (columns = characters, words):")
    print("  Matrix shape            :", result["shape"])
    print("  Column means (axis=0)   :", result["column_means"])
    print("  Column max   (axis=0)   :", result["column_max"])
    print("  Tickets above avg words :", result["above_average_words"])
    print("  Tickets over 12 words   :", result["long_tickets"])
    print("  Mean words per category :", result["mean_words_per_category"])


def make_charts(df: pd.DataFrame, results: list) -> list:
    """Create and save the three charts. Returns the list of saved file paths."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    # Chart 1: number of tickets in each category.
    counts = df["category"].value_counts().reindex(CATEGORIES, fill_value=0)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color="#2a78d6")
    plt.bar_label(bars)
    plt.title("Tickets per category")
    plt.xlabel("Category")
    plt.ylabel("Number of tickets")
    path = CHART_DIR / "01_category_counts.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()   # free memory before drawing the next chart
    saved.append(path)

    # Chart 2: histogram of how many words each ticket has.
    plt.figure(figsize=(8, 5))
    # One bar per word count (a bar for 4 words, one for 5 words, ...).
    # The edges sit halfway between whole numbers, e.g. 3.5, 4.5, 5.5 ...
    lowest = df["length_words"].min()
    highest = df["length_words"].max()
    bin_edges = np.arange(lowest - 0.5, highest + 1.5, 1)
    plt.hist(df["length_words"], bins=bin_edges, color="#2a78d6", edgecolor="white")
    median = df["length_words"].median()
    plt.axvline(median, color="black", linestyle="--", label=f"median = {median:.0f} words")
    plt.legend()
    plt.title("Ticket length distribution")
    plt.xlabel("Words per ticket")
    plt.ylabel("Number of tickets")
    path = CHART_DIR / "02_ticket_length_histogram.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    saved.append(path)

    # Chart 3: accuracy and macro-F1 of both models, side by side.
    names = [r["name"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    macro_f1 = [r["macro_f1"] for r in results]
    x = np.arange(len(names))   # bar positions: 0, 1
    width = 0.35
    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x - width / 2, accuracy, width, label="Accuracy", color="#2a78d6")
    bars2 = plt.bar(x + width / 2, macro_f1, width, label="Macro-F1", color="#eb6834")
    plt.bar_label(bars1, fmt="%.3f")
    plt.bar_label(bars2, fmt="%.3f")
    plt.xticks(x, names)
    plt.ylim(0, 1.15)
    plt.legend(loc="upper center", ncol=2)
    plt.title("Model scores on the same test set")
    plt.xlabel("Model")
    plt.ylabel("Score (0 to 1)")
    path = CHART_DIR / "03_model_scores.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    saved.append(path)

    return saved
