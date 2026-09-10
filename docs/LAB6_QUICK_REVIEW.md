# Lab 6 — Short assistant review

**Current workflow:** [follow the course GitHub steps](LAB6_GITHUB_WORKFLOW.md). Run `python scripts/review_lab6.py --limit 5` to review and save individual decisions without editing JSON.


**All 120 examples have been reviewed by the assistant. This is not a claim that you reviewed them.**

You do not need to fill 120 blank explanations to understand the result: the sample contains **45 distinct texts and only three recurring scenarios**, all labelled **parks** and predicted **roads**. Details for every original ID are saved in `artifacts/lab6/assistant_error_review.json`.

| Group | Examples | Representative source text | Assistant assessment |
|---|---:|---|---|
| Playground maintenance | 42 | ألعاب الأطفال في حديقة جدة تحتاج صيانة | Park play equipment needs maintenance; the topic is parks even when the park name contains a road or street name. |
| Park accessibility | 46 | الممر في حديقة جدة غير مناسب للكراسي المتحركة | A walkway inside a park is inaccessible to wheelchairs; keep the park context when interpreting the word walkway. |
| Park irrigation | 32 | الري متوقف في حديقة الرياض | Irrigation has stopped inside a park; the supplied course label is parks, not roads. |

## The main finding

Every text explicitly mentions a park (حديقة). The observed category is **topic confusion: parks → roads**, not 120 unrelated defects. The full supplied file has 300 errors, all in this same direction. We cannot tell from the prediction file alone whether the cause is model behaviour, label mapping, or deliberately constructed example predictions.

Thirteen sampled examples contain a hamza variant and/or elongated polite prefix. They remain understandable park requests. Those features are marked per entry, not claimed as causes. Road words in park/place names also deserve paired tests.

![Assistant error histogram](../artifacts/lab6/assistant_error_taxonomy.png)

## Three proposed fixes

1. **Verify the supplied prediction file and label-ID mapping against its generating model before diagnosing model internals.** Every one of the 300 supplied errors is parks → roads; the current trained classifier has zero saved validation errors. Expected improvement: Unknown until provenance is verified. Oracle scenarios quantify the maximum available correction, not a promised gain.
2. **Add paired parks-versus-roads probes covering park walkways and road words inside place names.** The park-accessibility group and road-named locations are direct candidates for testing context versus keyword shortcuts. Expected improvement: Unknown; these probes measure whether the hypothesized shortcut exists before any retraining.
3. **Compare clean/noisy versions of park complaints while holding the correct label fixed.** Hamza variants and elongated prefixes occur in the sample, but clean examples fail as well; normalize only after demonstrating a paired benefit. Expected improvement: Unknown; secondary spelling features are observations, not established causes.

## Quantifying possible correction without inventing results

The supplied file's macro-F1 is 0.8333. An oracle replacing only the sampled 120 wrong predictions with their gold labels would add 8.42 macro-F1 points. Correcting all 300 supplied errors would add 16.67 points. These are ceilings from label accounting, not measured improvements from a model fix. Nothing was overwritten.

## What remains for you

For understanding, read the three rows above and the spelling caveat. If you want this grouped assistant review accepted instead of the course’s human review, ask the instructor to approve that substitution. Reading three summaries does not automatically certify 120 individual human reviews.

You can send the instructor: “The supplied 120-error sample repeats three parks→roads scenarios. I have an explicitly labelled assistant review for every entry, a grouped summary, and a histogram. May I submit that with a representative human check instead of 120 separate handwritten annotations?”

The original human review fields remain unchanged. No requirement, expected label or source prediction was edited.
