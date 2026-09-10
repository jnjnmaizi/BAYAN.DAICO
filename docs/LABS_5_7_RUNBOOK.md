# Labs 5–7: implementation, results and remaining work

The code and measurements were produced locally on this Mac. `requirements.txt`, `pyproject.toml`, the course README/checklist, source data and original test contracts were not changed. Large trained models, FAISS indexes and ONNX files remain local under ignored `artifacts/` paths; GitHub contains code and measured JSON evidence.

## Lab 5 — Search

1. Fetch the two pinned multilingual checkpoints with `python scripts/setup_lab5.py`. The encoder represents both Arabic and English in one vector space; the cross-encoder scores each query/case pair together.
2. Run `python notebooks/05_retrieval_eval.py --device mps` on this Mac, or omit the flag for CPU. The first run builds all 20,000 cases. Later runs display the saved report. CPU encoding uses one thread because the installed macOS Torch/FAISS combination crashed with multiple CPU embedding threads.
3. The index stores L2-normalized vectors, case metadata and a manifest containing preprocessing/model versions and file checksums. Loading rejects incompatible or corrupted artifacts.
4. Evaluate the top 10 against the original relevant case IDs, with 50 candidates before reranking. The report includes recall, reciprocal rank, language comparisons, latency and empty-result calibration.
5. Compare the same raw vectors without normalization. **The expected metric collapse did not occur in this run:** raw-vector exact-ID scores were higher than the normalized first stage. We report this observation rather than inventing the expected failure. The production index still enforces unit norms.

Measured reranked recall@10 is **0.0231**, MRR@10 **0.0265**; both miss the targets. There are 5,401 unique texts among 20,000 cases, and each query names only three relevant case IDs. Semantically similar unlabelled cases count as wrong under the supplied contract. All answerable queries have both same-language and cross-language relevant IDs, so the report also compares those subsets within each query rather than treating Arabic-versus-English queries as a cross-lingual measure.

Empty-correct is **20/20**, while retaining all 130 answerable queries, on the calibration set. Those 20 no-answer rows contain **one unique text**. This is not independent generalisation evidence. See `artifacts/lab5/retrieval.json`, `data_audit.json` and `query_results.json`.

## Lab 6 — Evaluation

1. `python scripts/prepare_error_review.py` prepared a fixed seed-42 sample of 120 errors from the **supplied course predictions**. It refuses to overwrite an existing review. These are not errors from our current topic model, whose saved validation predictions are perfect.
2. Start with [the short assistant review](LAB6_QUICK_REVIEW.md): all 120 entries have assistant explanations and collapse into three recurring scenarios. These annotations do not count as human confirmation. If the instructor accepts a representative/grouped review, follow that agreed alternative; otherwise review [the worksheet](LAB6_ERROR_REVIEW.md). Record your chosen `category`, `reviewer_note`, and `human_confirmed: true` in `artifacts/lab6/human_error_review.json`. Only explicitly confirmed entries count; no automatic taxonomy tags are treated as human work.
3. `python scripts/sentiment_behaviour.py` trains a small TF-IDF/LinearSVC sentiment baseline on the existing training split. This is necessary to exercise the supplied sentiment-direction tests, because topic probabilities are not sentiment scores. All 200 checks pass through unchanged predictions; sentiment validation macro-F1 is only 0.3333, so this is a weak test result, not evidence of negation understanding.
4. `python scripts/evaluation_report.py` generates sliced metrics, confidence intervals, behavioural evidence, the evaluation report and three model cards. Add `--device mps` for initial behavioural inference on this Mac. Saved model reports are reused; changes to the human review are incorporated without repeating frozen-test inference.
5. The main topic report has 16 slices. The paired topic/DA comparison uses the same 1,200 Arabic validation rows. Confidence intervals resample citizen groups, not unrelated independent rows. Missing Gulf scores remain unavailable.
6. Topic invariance is **200/200**, MFT **15/16**. The optional Arabic DA model passes **8/8 Arabic MFT**, but only 1/8 English stress probes; its combined bilingual score is reported without concealing that intended-use difference.

The **120 human confirmations, confirmed-category histogram and review-grounded top-three fixes remain pending your review**. Suggested fixes in the report are clearly marked proposals; unknown future metric gains are not invented. See [EVALUATION_REPORT.md](../EVALUATION_REPORT.md) and [model cards](../model_cards).

## Lab 7 — CPU optimization and serving

The original CPU benchmark was run before ONNX export. Four CPU threads, 10 warm-up requests, and 200 seeded rows sampled without replacement from the supplied 2,000-row length mix were used consistently. Bare timings exclude tokenization; HTTP timings include it.

```bash
# Baseline first, for each task:
python scripts/benchmark_inference.py --task classifier
python scripts/benchmark_inference.py --task ner

# Export and retain fp32 rollback:
python scripts/export_onnx.py --task classifier
python scripts/export_onnx.py --task ner

# Repeat these for --task ner and --backend int8:
python scripts/benchmark_inference.py --task classifier --backend onnx
python scripts/check_onnx_quality.py --task classifier --backend onnx
python scripts/select_artifacts.py
```

Exports use dynamic input shapes and signed per-channel INT8 MatMul weights; embeddings stay fp32. This explains why file size reduction is smaller than the latency improvement. Both ONNX and INT8 agree with all saved classifier and NER validation predictions. No frozen-test inference was repeated.

The selected INT8 classifier has bare p99 **4.84 ms**, versus **187.39 ms** for padded fp32, a **38.7×** p99 speed-up. INT8 NER p99 is **4.86 ms**. The paired quality-tax intervals are zero on this synthetic validation set, not a universal quality guarantee.

```bash
source .venv/bin/activate
make serve
# In a second terminal:
bash scripts/load_test.sh
```

Startup canaries verify selected weights, tokenizer and label configuration hashes, preprocessing version, PII handling, and two labelled classifier probes. API routes are `/health`, `/v1/classify`, `/v1/entities`, `/v1/search`, and `/v1/analyse`. Entity offsets refer to the returned PII-masked source text; truncation is explicitly reported. Classification uses a bounded queue with at most 8 requests per microbatch and a 0.5 ms collection window, **without caching predictions**.

The HTTP test uses 16 persistent clients for 60 seconds with the supplied request text. A Python/httpx driver preserves measured JSON instead of requiring a separate `hey` installation. The original unbatched result remains in `artifacts/lab7/http_load_unbatched.json`; the final result is in `http_load.json` and `BENCHMARKS.md`.

## Final integration check

```bash
python -m pytest -q
```

All **140 tests passed** after implementing search, bootstrap, reports, ONNX selection, API integration and microbatching. This does not mean every numerical course target is achieved: see the status table below.

## Mandatory items still requiring evidence or human work

| Lab | Remaining condition |
|---|---|
| 3 | Topic +8 macro-F1 gain remains unmet: the original baseline already reaches the supplied test ceiling. |
| 4 | LOCATION +4 points remains unmet above a 100% original validation baseline. |
| 5 | Recall/MRR targets remain unmet; sparse exact-ID relevance judgements and repeated cases need course guidance/review. Normalization ablation did not demonstrate the expected collapse. |
| 6 | Human review 0/120 confirmed, category histogram and review-grounded prioritization pending. |
| 7 | Measured targets met: bare classifier p99 4.84 ms; 38.7× speed-up; zero validation quality tax; HTTP p99 33.36 ms at 16 clients with zero errors. |

## Optional work deferred as requested

- Further Arabic model bake-off or MARBERT training. The updated README marks this extra practice; earlier Mix/DA results are retained.
- Further attempts to establish the optional Gulf model gain without unseen Gulf data. No Gulf validation examples exist in the current split.
- Additional models or hyperparameter sweeps beyond the mandatory comparisons.
- Capstone extensions beyond assembling the existing lab components; no new extension project was started.

Do not change the original expected values, labels or requirements to force a pass. Report unachieved targets with their evidence and ask the instructor how to handle the dataset limitations.

Model sources: [multilingual encoder](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) and [multilingual reranker](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1). Exact revisions are pinned in `artifacts/lab5/model_sources.json`.
