"""Simple checks for TicketTriage. Run with:  python test_basic.py

No extra libraries needed. Each test uses plain `assert`.
If an assert is false, Python stops and shows which test failed.
"""

from pathlib import Path
import tempfile

from analytics import load_dataset, make_charts
from api_client import ProductAPIClient
from exceptions import InvalidMessageError, TicketTriageError
from manager import TicketManager
from models import NaiveBayesClassifier, TicketClassifier, train_and_compare
from text_utils import clean_text


def test_clean_text():
    assert clean_text("Hi, I have NOT received my parcel!") == "have not received my parcel"
    assert clean_text("Visit https://x.com or mail a@b.com 123") == "visit mail"
    assert clean_text(None) == ""   # empty CSV cells must not crash


def test_load_dataset():
    df, info = load_dataset()
    # All five categories are present and only those five.
    assert sorted(df["category"].unique()) == ["Account", "Delivery", "Order", "Payment", "Refund"]
    assert df["instruction"].isna().sum() == 0
    assert info["rows_out_of_scope"] > 0   # CONTACT/INVOICE/FEEDBACK rows were removed


def test_abstract_class_and_untrained_model():
    # An abstract class cannot be created directly.
    try:
        TicketClassifier()
        assert False, "TicketClassifier should not be creatable"
    except TypeError:
        pass
    # Predicting before training gives our own error, not a crash.
    try:
        NaiveBayesClassifier().predict(["hello"])
        assert False, "should have raised TicketTriageError"
    except TicketTriageError:
        pass


def test_models_and_manager():
    df, _ = load_dataset()
    best_model, results, _ = train_and_compare(df)
    assert len(results) == 2
    for result in results:
        assert 0 <= result["accuracy"] <= 1 and 0 <= result["macro_f1"] <= 1
    # The selected model really has the highest macro-F1.
    assert max(r["macro_f1"] for r in results) == next(
        r["macro_f1"] for r in results if r["name"] == best_model.name)

    manager = TicketManager(best_model)
    prediction = manager.classify("I forgot my password and cannot log in")
    assert prediction["category"] == "Account"
    assert manager.route("Refund") == "Returns & Refunds Team"

    for bad in ["", "   ", "ab", "12345 !!!"]:
        try:
            manager.classify(bad)
            assert False, f"'{bad}' should be rejected"
        except InvalidMessageError:
            pass

    predictions, skipped = manager.classify_csv(Path(__file__).parent / "data" / "sample_batch.csv")
    assert len(predictions) == 11 and len(skipped) == 1   # one blank row in the sample

    paths = make_charts(df, results)
    assert len(paths) == 3 and all(p.is_file() for p in paths)


def test_api_offline_fallback():
    # Port 9 on this computer has nothing listening, so the "live" call fails at once,
    # just like being offline.
    with tempfile.TemporaryDirectory() as folder:
        cache = Path(folder) / "product_1.json"
        client = ProductAPIClient(url="http://127.0.0.1:9/products/1", cache_file=cache)
        product, source = client.get_product()
        assert source == "bundled sample" and product["id"] == 1

        cache.write_text('{"id": 1, "title": "Cached product"}', encoding="utf-8")
        product, source = client.get_product()
        assert source == "cache" and product["title"] == "Cached product"


if __name__ == "__main__":
    tests = [test_clean_text, test_load_dataset, test_abstract_class_and_untrained_model,
             test_models_and_manager, test_api_offline_fallback]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
