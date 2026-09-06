# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1
- Class: Tatweel / decorative elongation
- Example: `الخدمــــة` and `مـشكلة`
- Why it matters: these are the same lexical forms but create different token sequences.
- Decision: clean — remove U+0640 tatweel before tokenisation.

### Defect 2
- Class: Repeated-character emphasis
- Example: `لووووسمحت` and English `pleaseeee`
- Why it matters: expressive elongation inflates sequence length and vocabulary sparsity.
- Decision: clean conservatively — reduce runs of 3+ to two characters, retaining emphasis.

### Defect 3
- Class: Irregular whitespace and line breaks
- Example: leading/trailing spaces in `The fees for Balady are unclear`, plus `<br>`-adjacent spacing.
- Why it matters: equivalent feedback should not tokenize differently because of layout.
- Decision: clean — collapse all Unicode whitespace and trim.

### Defect 4
- Class: Saudi phone and national-ID-shaped PII
- Example: `0551234567`, `+966551234567`, and `1023456789` in feedback rows 1, 38, and 42.
- Why it matters: PII must not reach training, evaluation, or serving logs in raw form.
- Decision: clean — replace with `<PHONE>` and `<NATIONAL_ID>` before normalisation.

### Defect 5
- Class: Arabic/English code-switching and product identifiers
- Example: `BYN-2026-000003`, English service names within Arabic feedback, and Arabic location names.
- Why it matters: bilingual model selection must preserve service and reference context.
- Decision: preserve — tokenizers should learn these forms; no transliteration or lowercasing.

### Defect 6
- Class: Emoji and HTML remnants
- Example: `😡` and terminal `<br>` (for example rows 1, 32, and 63).
- Why it matters: emoji is sentiment signal, whereas the HTML fragment is source-system noise.
- Decision: task-dependent — preserve emoji in the shared normaliser; remove/parse HTML only in a separately versioned source-ingestion rule if required.

### Sentence-segmentation spot check
- Arabic multi-sentence complaint: `الماء منقطع منذ الصباح. أرجو إرسال فني اليوم؟ شكراً!` split into 3 sensible sentences.
- English multi-sentence complaint: a `p.m.` abbreviation stayed joined to `yesterday.`; the example split into 3 sentences.
- Numbered-list complaint: `1. الإنارة ... 2. توجد حفرة ... 3. الحاوية ...` yielded one complete sentence per list item.
- Reference/abbreviation example: `Dr. Ali` remained together; the reference sentence and follow-up were split separately.
- Arabic emphatic complaint: `!!!` is conservatively normalised to `!!`, then `؟` and `.` create sensible boundaries.

## Lab 2 — Parameter audit
| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | | | |
| CAMeLBERT | | | |

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
