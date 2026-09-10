# First unfinished item — Lab 1 sequence-length histogram

The missing histogram has now been generated from the same 12,000 raw feedback rows used by the original audit. The script uses pinned tokenizer revisions, keeps the data unchanged and measures full sequences without truncation. No model training or GPU is needed.

## 1. Open the VS Code terminal

Use Terminal → New Terminal. Enter:

```bash
cd /Users/j.alhumaizi/Downloads/BAYAN.DAICO-main
source .venv/bin/activate
```

The environment contains the installed project libraries. Keep using this terminal for the next commands.

## 2. Run the audit

```bash
python notebooks/01_tokenizer_audit.py
```

The script reads 7,200 Arabic and 4,800 English rows. It loads each tokenizer from the local cache first; a fresh machine may need to download the pinned tokenizer files. It prints the comparison table and writes two outputs:

- [token_length_histograms.png](../artifacts/lab1/token_length_histograms.png): the chart with four panels.
- [tokenizer_audit.json](../artifacts/lab1/tokenizer_audit.json): exact frequency counts, statistics, tokenizer revisions and the source-data hash.

Expected p95 values:

| Tokenizer | Arabic | English |
|---|---:|---:|
| mBERT | 27 | 25 |
| XLM-R | 21 | 23 |
| CAMeLBERT | 20 | 38 |
| DistilBERT | 47 | 21 |

The last two terminal messages start with `Saved chart:` and `Saved numeric results:`. If they do not appear or you get a traceback, send the full output.

## 3. Open and read the chart

In VS Code's Explorer, open `artifacts` → `lab1` → `token_length_histograms.png`, or click the image link above.

- Each panel is a tokenizer.
- Blue represents Arabic and orange represents English.
- The horizontal axis is the number of tokens per feedback text, including special tokens such as the model's sentence-boundary markers. Tokens are subword units, not necessarily whole words.
- The vertical axis is the percentage of feedback rows **within that language** at a given length. Each language totals 100%, which makes comparison fair despite the different row counts.
- Dashed lines mark p95. For XLM-R, Arabic p95 is 21 and English p95 is 23: at least 95% of the respective observed sequences are no longer than these values.

A distribution shifted to the right means the same corpus needs more tokens. That can increase processing cost and truncation risk for a fixed token limit. It does not directly measure model accuracy.

## 4. Explain the decision

The measured justification is:

> XLM-R gives a balanced Arabic/English token-length profile: p95 is 21/23. CAMeLBERT is slightly shorter for Arabic (20) but much longer for English (38). DistilBERT is short for English (21) but much longer for Arabic (47). This supports choosing XLM-R for the shared bilingual pipeline.

The histogram supplements the existing fertility and p95 table. It does not justify setting every model's maximum length to p95: some inputs are longer, and QA contexts are a different input distribution.

## 5. Check Lab 1's original tests

```bash
python -m pytest tests/test_preprocessing.py tests/test_pii_recall.py -q
```

Expected: **26 passed** (25 preprocessing cases plus the PII test that checks all 60 examples).

The generated statistics match the earlier recorded table. All eight histogram distributions were checked for the correct row totals and p95 values. This closes the missing Lab 1 histogram deliverable. We will handle the Lab 3 improvement-target issue next, after you have run and inspected this output.
