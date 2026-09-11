<div align="center">

# JanaAlhumaizi.NLP

### Bilingual feedback. Structured insight. Measured performance.

An Arabic and English NLP prototype that turns citizen feedback into service topics, named entities, and searchable historical cases.

**Arabic + English · Transformer models · Semantic search · FastAPI · ONNX INT8**

[Architecture](#how-it-works) · [Results](#measured-results) · [Run locally](#run-locally) · [Technical evidence](#explore-the-project)

</div>

---

## The opportunity

Citizen-service teams receive free-text complaints that vary in language, spelling, and detail. Reading each message, identifying the responsible service, and locating related cases takes time and makes consistent handling difficult.

**JanaAlhumaizi.NLP demonstrates how one NLP service can support that workflow.** It prepares bilingual text, masks supported personal identifiers, predicts one of eight service topics, extracts useful entities, and searches a corpus of 20,000 historical cases. A separate extractive question-answering component selects answers from supplied context.

**JanaAlhumaizi.NLP** is an independent project inspired by the [SDAIA Academy BAYAN project](https://github.com/AljawharaAlbahlalDev/SDA-AIE-211-Bayan-Course), with project-specific preprocessing, model, evaluation, serving and optimisation modifications. It was built by **Jana Alhumaizi** for **SDA-AIE-211 — Natural Language Processing with Transformers**. Current status: **working educational prototype evaluated on synthetic data**.

## Value for a service team

| Business need | What the project provides | Intended benefit |
|---|---|---|
| Organise incoming feedback | Topic classification across billing, digital services, licensing, lighting, parks, roads, waste, and water | Support consistent triage and routing. |
| Understand the important details | Named entity recognition (NER) for locations, dates, references, and services | Help reviewers find key facts without reading every sentence repeatedly. |
| Handle Arabic and English together | Shared multilingual models and explicit Arabic preprocessing profiles | Support a common workflow across both languages. |
| Find related historical cases | Vector retrieval followed by a multilingual reranker | Surface potentially relevant cases for a reviewer. |
| Limit exposure of supported identifiers | Phone and national-ID masking before model use | Reduce exposure of these identifiers in the processing pipeline. |
| Integrate with existing systems | Typed HTTP endpoints, model manifests, and startup checks | Provide a defined interface for a future portal or case-management integration. |

The workflow is designed to help service teams organise feedback, inspect key details, and explore related cases through one interface.

## Project highlights

- **One bilingual workflow:** Arabic and English feedback share classification, entity extraction, and search interfaces.
- **Fast CPU execution:** the optimised classifier achieved **4.84 ms p99 model latency** in the recorded benchmark.
- **Connected analysis:** one API request combines a topic prediction, extracted entities, and historical-case candidates.
- **Traceable model delivery:** pinned checkpoints, versioned preprocessing, artifact checksums, and startup checks make the deployed components identifiable.
- **Evidence-backed optimisation:** saved timing runs and paired FP32/INT8 comparisons support the performance results below.
- **Complete Lab 6 evidence:** all 120 sampled errors have been individually reviewed, with the grouped summary retained for traceability.

## How it works

```mermaid
flowchart LR
    A[Arabic or English feedback] --> B[Mask supported identifiers]
    B --> C[Task-specific text preparation]
    C --> D[Topic classifier]
    C --> E[Entity extraction]
    C --> F[Vector retrieval]
    F --> G[Cross-encoder reranking]
    D --> H[Structured API response]
    E --> H
    G --> H
    H --> I[Reviewer or client application]

    classDef input fill:#eff6ff,stroke:#2563eb,color:#0f172a
    classDef model fill:#f0fdfa,stroke:#0d9488,color:#0f172a
    classDef output fill:#f5f3ff,stroke:#7c3aed,color:#0f172a
    class A,B,C input
    class D,E,F,G model
    class H,I output
```

The combined `/v1/analyse` endpoint assembles classification, entities, and search. Extractive QA runs through dedicated lab scripts. Entity offsets refer to the returned PII-masked text.

## Tools and why they were chosen

| Technology | Role | Reason for the choice |
|---|---|---|
| **Python 3.12, PyTorch, Hugging Face Transformers** | Training and model inference | A shared workflow for pretrained checkpoints, fine-tuning, token alignment, and export. |
| **XLM-R** | Bilingual classification and NER | The tokenizer audit showed a balanced Arabic/English representation; one backbone supports both tasks. |
| **CAMeL Tools and CAMeLBERT** | Arabic normalisation, clitic segmentation, and Arabic model experiments | Make Arabic-specific preprocessing choices explicit and measurable. |
| **scikit-learn** | TF-IDF baseline and evaluation | Provides a transparent baseline against which to assess transformer results. |
| **Multilingual MiniLM and FAISS** | Case embeddings and vector search | Retrieve candidates from a shared Arabic/English vector space using normalised similarity. |
| **Multilingual cross-encoder** | Rerank retrieved cases | Score the query and case together after the initial candidate search. |
| **ONNX Runtime and INT8 quantisation** | CPU inference | Reduce measured inference latency while checking agreement against saved FP32 predictions. |
| **FastAPI and Pydantic** | HTTP service and input validation | Expose predictable request/response interfaces with interactive API documentation. |
| **pytest, bootstrap evaluation, Matplotlib** | Verification and reporting | Check contracts, quantify uncertainty, and make measured evidence inspectable. |

Exact model revisions, preprocessing versions, and artifact checksums are recorded in the [benchmark evidence](BENCHMARKS.md) and model manifests.

## Measured results

![CPU p99 latency across PyTorch, ONNX FP32, and ONNX INT8 for classification and entity extraction](docs/assets/inference_latency.png)

The chart uses saved runs on an **Apple Silicon CPU, four threads, batch size one, 200 sampled inputs, and 10 warm-ups**. It measures model execution only. The padded baseline uses 512 tokens; dynamic runs cap inputs at 128. The total improvement therefore includes both padding changes and runtime optimisation.

| Evidence | Recorded result | Scope |
|---|---|---|
| Classifier model latency | **4.84 ms p99** with INT8 | 38.7× faster than the padded PyTorch baseline; 7.4× faster than dynamic PyTorch. |
| Entity model latency | **4.86 ms p99** with INT8 | Same sampled-input benchmark protocol. |
| Classification HTTP load | **33.36 ms p99**, about **554 requests/second** | 16 concurrent clients over 60 seconds; 33,259 requests, zero errors; one fixed payload. |
| INT8 prediction agreement | **100% agreement with saved FP32 predictions** | 2,400 classifier and 935 NER validation examples; no observed quality loss on those examples. |
| Supported PII masking | **60/60 test cases** | Verified against the supplied phone and national-ID test cases. |
| Search corpus | **20,000 cases** | Full corpus indexed with normalised vectors, metadata, and a versioned manifest. |

Sources: [classifier timing](artifacts/lab7/classifier_int8_benchmark.json), [NER timing](artifacts/lab7/ner_int8_benchmark.json), [HTTP load](artifacts/lab7/http_load.json), [classifier quality](artifacts/lab7/classifier_int8_quality.json), [NER quality](artifacts/lab7/ner_int8_quality.json), and [full benchmarks](BENCHMARKS.md).

Results describe the recorded prototype runs on synthetic course data and the hardware specified above. Full metric definitions and evaluation context are available in the [technical evaluation report](EVALUATION_REPORT.md).

## Secondary external evaluation

The repository also contains a separate external-data track for coverage gaps
in the supplied course evaluation. It does not modify the official Bayan data,
labels, requirements, or scores.

| External benchmark | Purpose | Recorded result |
|---|---|---:|
| [ArBNTopic](https://huggingface.co/datasets/U4RASD/ArBNTopic) | 14-class Arabic topic coverage | Accuracy **0.740366**, macro-F1 **0.719714** on 1,583 test rows |
| [Alyah](https://huggingface.co/datasets/tiiuae/alyah-emirati-benchmark) | Emirati/Gulf dialect coverage | 1,173 manually curated test questions |
| [IAHLT Arabic NER](https://huggingface.co/datasets/iahlt/arabic_ner_mafat) | Independent Arabic location-entity coverage | 1,179 location-like spans in a 2,000-row sample |
| [ArabicRAGB](https://huggingface.co/datasets/HeshamHaroon/ArabicRAGB) | Query-to-positive-passage retrieval | Recall@10 **0.946000**, MRR@10 **0.829494** on 500 sampled queries |

These are independent benchmarks with different label sets or relevance
definitions. They provide additional evidence and do not replace the official
course metrics. Reproduce them with the [Colab notebook](notebooks/secondary_evaluation.ipynb)
or the scripts in `scripts/secondary_*.py`. The full explanation is in the
[secondary results report](docs/SECONDARY_RESULTS_2026-09-11.md).

## Evaluation notes and limitations

- **Synthetic data:** the course datasets are synthetic and contain repeated
  templates, so the measured scores describe this benchmark rather than broad
  real-world service performance.
- **Lab 3 coverage:** the frozen topic test contains only four of the eight
  training labels. Both models reach 100% accuracy and the fixed eight-label
  macro-F1 ceiling is 0.5000, leaving no available headroom for the requested
  improvement.
- **Lab 4 Gulf coverage:** the supplied validation split contains no Gulf rows,
  so a Gulf generalisation delta cannot be measured. The original LOCATION
  recall is already 1.0000 on its validation set, so a positive four-point
  improvement is not possible on that split.
- **Lab 5 relevance labels:** the official retrieval score uses the supplied
  exact case IDs. The audit found that the 130 answerable queries use IDs that
  follow the repeating eight-topic corpus cycle, while repeated case text can
  have different IDs. Semantically or textually correct results can therefore
  count as incorrect under the official labels.
- **Official versus secondary metrics:** the official Bayan metrics remain
  unchanged. External benchmark results are reported separately because their
  label sets, tasks, or relevance definitions differ.
- **Lab 6 review scope:** the 120 sampled errors were individually confirmed;
  the 45 grouped decisions are retained as a compact traceability summary.
- **Lab 7 performance scope:** latency was measured on one Apple Silicon
  machine using fixed payloads, batch size one, four threads and the recorded
  concurrency. It does not guarantee the same latency for every payload,
  hardware platform or production workload.
- **Runtime artifacts:** large model checkpoints, FAISS files and pinned
  encoder/reranker assets are excluded from Git. A clean clone needs those
  artifacts prepared before the full API and search service can start.
- **External-data licensing:** the secondary scripts download public dataset
  releases at runtime. Raw external records are not redistributed in this
  repository; review each dataset card and licence before reuse.
- **No source rewriting:** no official labels, requirements, frozen-test rows
  or failed official scores were changed to create a numeric pass.

## Run locally

From the project directory, create the required Python environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

**Starting the API requires the local trained models, selected ONNX exports, and their manifests. Search also requires the FAISS index and pinned encoder/reranker checkpoints.** Large runtime artifacts are excluded from Git; installing dependencies alone does not create them. Follow the [training walkthrough](docs/LAB3_WALKTHROUGH.md) and [Labs 5–7 runbook](docs/LABS_5_7_RUNBOOK.md) to prepare them.

With those artifacts available:

```bash
python -m uvicorn bayan.serving.api:app --host 127.0.0.1 --port 8000
```

Open [interactive API documentation](http://127.0.0.1:8000/docs), or submit a request:

```bash
curl -X POST http://127.0.0.1:8000/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"There is a pothole in the road near the park."}'
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check startup validation and selected model information. |
| `POST /v1/classify` | Return the predicted topic and model score. |
| `POST /v1/entities` | Extract entity spans from the masked text. |
| `POST /v1/search` | Return matching historical cases above the configured threshold. |
| `POST /v1/analyse` | Combine classification, entity extraction, and search. |

All POST endpoints accept a JSON object with a `text` field.

## Explore the project

| Area | Where to look |
|---|---|
| Preprocessing and Arabic handling | [Source](src/bayan/preprocessing/) · [Tokenizer audit chart](artifacts/lab1/token_length_histograms.png) |
| Models, attention, and training | [Models](src/bayan/models/) · [Attention implementation](src/bayan/attention.py) · [Scripts](scripts/) |
| Retrieval and serving | [Search](src/bayan/search/) · [API](src/bayan/serving/api.py) |
| Results and design decisions | [Benchmarks](BENCHMARKS.md) · [Evaluation report](EVALUATION_REPORT.md) · [Decisions](DECISIONS.md) |
| Model documentation | [Model cards](model_cards/) |
| Course instructions | [Lab checklist](docs/LABS.md) · [Original course README](https://github.com/AljawharaAlbahlalDev/SDA-AIE-211-Bayan-Course/blob/7949de02de71cd3ae644cd89f766dfc73aa9b7f4/README.md) |

To regenerate the performance chart from the saved measurements:

```bash
python scripts/render_readme_benchmarks.py
```

## Acknowledgements

Developed from the [SDA-AIE-211 Bayan course repository](https://github.com/AljawharaAlbahlalDev/SDA-AIE-211-Bayan-Course). Original course materials and contributions remain attributed to their authors. See [LICENSE.txt](LICENSE.txt) for the supplied data notice.

**SDAIA Academy GitHub:** [github.com/SDAIAAcademy](https://github.com/SDAIAAcademy)
