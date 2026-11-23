# TicketTriage — AI Customer Support Ticket Classifier

A console program for a small online shop. It reads a short customer message and suggests
one of five categories (**Account, Order, Delivery, Payment, Refund**) and the support team
that should handle it. It also saves a prediction history, analyses the data with
Pandas/NumPy, draws three charts, and can look up a product from the DummyJSON API.

**Suggestions only:** the program never contacts customers or changes real records.
Tickets the model is unsure about (below 50% confidence) go to a *Human Review Queue*.

**Team:** Mohib Ul Haque (25K-0091) · Hassan Ul Haq (25K-0161)

## Setup (Python 3.10+)

```bash
cd TicketTriage
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**Dataset.** The program works straight away with the bundled **synthetic** file
`data/sample_tickets.csv`. For real results, download the Bitext CSV from
<https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset>
(Files tab → the `.csv` file) and save it as `data/bitext_customer_support.csv`.
If that file exists, the program uses it automatically.

## Run

```bash
python main.py         # start the program
python test_basic.py   # run the 5 automatic checks
```

| Option | What it does |
|---|---|
| 1 | Classify one typed message, then optionally save it to `outputs/history.json` |
| 2 | Classify every row of a CSV with a `message` column, save to `outputs/batch_results_*.csv` |
| 3 | Show the last 10 saved predictions |
| 4 | Pandas + NumPy analysis, and save 3 charts to `outputs/charts/` |
| 5 | Look up product 1 from DummyJSON (live → cached copy → bundled sample) |
| 6 | Compare both models' accuracy and macro-F1 and show which one was chosen |
| 0 | Exit |

### Sample session (synthetic data)

```text
Data: sample_tickets.csv [SYNTHETIC SAMPLE]
  250 rows -> 234 usable (removed 3 missing/empty, 12 other categories, 1 duplicates)
Training Naive Bayes and Logistic Regression...
  Selected: Logistic Regression

Choose an option: 1
Type the customer message:
> I forgot my password and cannot log in
  Category   : Account
  Confidence : 85.9%
  Route to   : Customer Accounts Team
Save to history? (y/n): y
  Saved. History now has 1 prediction(s).

Choose an option: 1
> what is the weather in karachi
  Category   : Order
  Confidence : 31.1%
  Route to   : Human Review Queue
  ! Low confidence: a human should check this ticket.

Choose an option: 6
Stratified split: 187 training rows, 47 test rows (random seed 42)
Model                   Accuracy  Macro-F1
Naive Bayes                0.936     0.931
Logistic Regression        0.936     0.934  <- selected
```

## Files

| File | What it contains |
|---|---|
| `main.py` | The menu and one function per menu option |
| `text_utils.py` | `clean_text()`: lowercase, remove noise, remove stopwords |
| `models.py` | `TicketClassifier` (abstract), `NaiveBayesClassifier`, `LogisticTicketClassifier`, evaluation and model selection |
| `manager.py` | `TicketManager`: validate, classify, route to team, flag low confidence, save history/CSV |
| `api_client.py` | `ProductAPIClient`: DummyJSON lookup with cache and offline fallback |
| `analytics.py` | Load and clean the dataset, Pandas/NumPy analysis, 3 charts |
| `exceptions.py` | `TicketTriageError`, `InvalidMessageError`, `APIUnavailableError` |
| `test_basic.py` | 5 simple automatic checks (plain `assert`, no extra libraries) |
| `data/` | `sample_tickets.csv`, `sample_batch.csv` (both synthetic), `product_sample.json` |

## How the model is built

1. **Clean the data** (`analytics.load_dataset`): drop rows with missing text or category,
   keep only the categories we need, and drop duplicate messages.
2. **Category mapping:** ACCOUNT→Account, ORDER→Order, DELIVERY→Delivery,
   **SHIPPING→Delivery** (the same team handles address changes), PAYMENT→Payment,
   REFUND→Refund. Other Bitext categories (CONTACT, INVOICE, FEEDBACK, NEWSLETTER,
   CANCELLATION_FEE) are removed.
3. **Clean each message** (`clean_text`): lowercase; remove links, e-mails, numbers and
   punctuation; remove our own stopword list. Words like "not" and "where" are kept.
4. **TF-IDF** turns text into numbers. It learns its vocabulary from the **training data
   only**, so the test data stays unseen.
5. **Stratified split** 80% train / 20% test, with `random_state=42`.
6. **Both models** are trained on the same training data and scored on the same test data
   using **accuracy** and **macro-F1**. The higher macro-F1 wins.

## Results

| Data | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| Synthetic sample (47 test rows) | Naive Bayes | 0.936 | 0.931 |
| Synthetic sample (47 test rows) | **Logistic Regression** (selected) | 0.936 | 0.934 |
| Bitext full dataset | *fill in after running option 6* | | |

The sample scores are **not** a real measure of quality. The data is generated from
templates, and with only 47 test rows each ticket changes accuracy by about 2%.

## Manual testing checklist

- Option 1 with a normal message → category and team are shown.
- Option 1 with an empty message, `ab`, or `!!!` → a clear error, then back to the menu.
- Option 1 with an off-topic message (e.g. weather) → "Human Review Queue".
- Option 2 + Enter → 11 classified, 1 skipped (the blank row 9). A CSV appears in `outputs/`.
- Option 2 with a wrong path → "File not found".
- Option 3 → the saved predictions are listed.
- Option 4 → 3 PNG files in `outputs/charts/`.
- Option 5 with Wi-Fi off → "bundled sample" (or "cache" if it worked online before).
- Option 6 → both models' scores and the selected model.
- Ctrl+C at the menu → "Goodbye." with no error text.

## Limitations

- Bitext and the sample are generated text. Real customers write longer, messier and
  mixed-language messages, so real accuracy will be lower.
- Each ticket gets one category, even when it is about two things (e.g. paid but not delivered).
- TF-IDF ignores word order beyond word pairs.
- The 50% confidence threshold is a simple rule of thumb.
- The models are retrained every time the program starts (a few seconds).

## Ethics

The program only suggests. People make the decisions. Unsure tickets are sent to human
review instead of being routed automatically. The history file is plain JSON on your
computer, so don't store real customer personal data in it without permission.

## Attribution

- **Dataset:** Bitext Customer Support LLM Chatbot Training Dataset (Hugging Face),
  licence **CDLA-Sharing-1.0**. It is not included in this repo.
- **Product API:** [DummyJSON](https://dummyjson.com). `data/product_sample.json` is a
  hand-written offline example in the same format.
- `data/sample_tickets.csv` and `data/sample_batch.csv` are **synthetic** demo files,
  not real customer messages.
- Libraries: pandas, NumPy, scikit-learn, Matplotlib, requests.

## Course concepts used

| Concept | Where |
|---|---|
| Core Python | menu loop, if/elif, input checks, dictionaries, f-strings, `pathlib` |
| Functions / lambda | one function per menu option and per cleaning step; `sorted(..., key=lambda pair: pair[1])` in batch mode |
| OOP | `TicketClassifier` ABC and 2 subclasses (inheritance, overriding), protected `_model` (encapsulation), `evaluate_model()` (polymorphism), `TicketManager`, `ProductAPIClient` |
| Files / APIs / exceptions | CSV read/write, JSON history and cache, `requests` with status check and timeout, 3 custom exceptions |
| NumPy | 2-D matrix, `axis=0` mean/max, slicing `matrix[:, 1]`, boolean masks, broadcasting `matrix - column_means` |
| Pandas | `dropna`, `fillna`, `isin` filtering, `map`, `drop_duplicates`, `describe`, `groupby().agg()` |
| Visualization | 3 labelled charts: category bars, length histogram, model scores |
| ML | stratified split, Naive Bayes vs Logistic Regression, accuracy, macro-F1, automatic selection |
| NLP / TF-IDF | own cleaning and stopwords, `TfidfVectorizer` with word pairs |
