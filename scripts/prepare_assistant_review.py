"""Publish assistant-reviewed Lab 6 annotations separately from human sign-off.

The group assignments below were made after reading all 120 sampled texts.
This script materializes that review; it neither trains a model nor changes labels.
"""
import json
from collections import Counter
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from bayan.models.training import ROOT, sha256, write_json

GROUPS = {
    'playground_maintenance': [5,7,8,10,11,12,14,15,21,23,24,25,35,37,38,46,52,53,58,59,61,63,64,72,74,75,78,79,83,84,86,91,92,94,95,97,100,101,103,108,110,115],
    'park_accessibility': [1,2,4,6,9,16,19,22,28,29,33,34,36,40,41,42,43,44,47,48,50,51,55,57,60,65,69,73,76,80,81,82,85,87,88,96,98,102,104,105,106,109,113,116,118,119],
    'park_irrigation': [3,13,17,18,20,26,27,30,31,32,39,45,49,54,56,62,66,67,68,70,71,77,89,90,93,99,107,111,112,114,117,120],
}
DETAILS = {
    'playground_maintenance': {
        'title':'Playground maintenance','evidence':'الأطفال',
        'rationale':'The text requests maintenance of children\'s play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it.',
        'check':'Park play equipment needs maintenance; the topic is parks even when the park name contains a road or street name.',
    },
    'park_accessibility': {
        'title':'Park accessibility','evidence':'للكراسي المتحركة',
        'rationale':'The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint.',
        'check':'A walkway inside a park is inaccessible to wheelchairs; keep the park context when interpreting the word walkway.',
    },
    'park_irrigation': {
        'title':'Park irrigation','evidence':'متوقف في حديقة',
        'rationale':'The text reports stopped irrigation within a named park. The explicit park asset supports the supplied parks label; roads is unsupported. Service ownership could distinguish parks/water in another taxonomy, but it does not justify roads here.',
        'check':'Irrigation has stopped inside a park; the supplied course label is parks, not roads.',
    },
}
PRIMARY_CATEGORY = 'Topic confusion: parks → roads'


def build_review(rows):
    if len(rows)!=120:
        raise ValueError('Assistant annotations require the original 120-row sample')
    mapping={number:key for key,indices in GROUPS.items() for number in indices}
    if len(mapping)!=120 or set(mapping)!=set(range(1,121)):
        raise ValueError('Review groups must cover all rows exactly once')
    reviews=[]
    for number,row in enumerate(rows,1):
        group=mapping[number]; details=DETAILS[group]; text=row['text']
        if row['y_true']!='parks' or row['y_pred']!='roads' or details['evidence'] not in text or 'حديقة' not in text:
            raise ValueError(f'Review no longer matches source row {number}')
        flags=[]; extra=[]
        if any(word in text for word in ['ألعأب','ألممر','ألري']):
            flags.append('orthographic_variation')
            extra.append('A visible hamza/spelling variant is present, but the park context remains explicit; its causal effect is untested.')
        if 'لووووسمحت' in text:
            flags.append('elongated_polite_prefix')
            extra.append('The elongated polite prefix adds surface noise without changing the requested service.')
        if any(word in text for word in ['شارع التحلية','طريق الملك فهد']):
            flags.append('road_word_in_place_name')
            extra.append('The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.')
        if '😡' in text: flags.append('emoji')
        if '<PHONE>' in text or '<NATIONAL_ID>' in text: flags.append('masked_pii_placeholder')
        if text!=text.strip(): flags.append('trailing_whitespace')
        reviews.append({'feedback_id':row['feedback_id'],'sample_number':number,'text':text,
            'y_true':row['y_true'],'y_pred':row['y_pred'],'assistant_category':PRIMARY_CATEGORY,
            'assistant_group':group,'evidence_phrase':details['evidence'],
            'assistant_note':details['rationale']+(' '+' '.join(extra) if extra else ''),
            'observed_features':flags,'label_assessment':'Supplied parks label supported by explicit park context',
            'cause_status':'Observed label confusion; model-internal cause and prediction-file provenance not established',
            'reviewer_type':'assistant','human_confirmed':False})
    return reviews


def main():
    out=ROOT/'artifacts/lab6'; source=out/'human_error_review.json'
    rows=json.loads(source.read_text()); reviews=build_review(rows)
    write_json(out/'assistant_error_review.json',reviews)
    supplied=ROOT/'data/eval/validation_predictions.csv'
    frame=pd.read_csv(supplied); errors=frame[frame.y_true!=frame.y_pred]
    labels=sorted(frame.y_true.unique())
    before=float(f1_score(frame.y_true,frame.y_pred,labels=labels,average='macro',zero_division=0))
    # Oracle accounting scenarios, never a model or an edit to the source file.
    def scenario(ids):
        corrected=frame.y_pred.copy(); mask=frame.feedback_id.isin(ids); corrected.loc[mask]=frame.loc[mask,'y_true']
        return {'corrected_rows':int(mask.sum()),'macro_f1':float(f1_score(frame.y_true,corrected,labels=labels,average='macro',zero_division=0)),
            'macro_f1_gain_points':float(100*(f1_score(frame.y_true,corrected,labels=labels,average='macro',zero_division=0)-before)),
            'accuracy_gain_points':float(100*(accuracy_score(frame.y_true,corrected)-accuracy_score(frame.y_true,frame.y_pred)))}
    summary={'reviewer_type':'assistant','assistant_reviewed':len(reviews),'human_confirmed_in_original_file':sum(r.get('human_confirmed') is True for r in rows),
        'official_human_requirement_satisfied_by_assistant_review':False,
        'source_predictions_sha256':sha256(supplied),'source_sample_sha256':sha256(source),
        'source_scope':'Supplied course predictions, not errors from the current trained topic classifier',
        'primary_category_histogram':dict(Counter(r['assistant_category'] for r in reviews)),
        'scenario_histogram':dict(Counter(r['assistant_group'] for r in reviews)),
        'feature_counts':dict(Counter(flag for row in reviews for flag in row['observed_features'])),
        'unique_sample_texts':len({r['text'] for r in reviews}),
        'full_source_error_count':len(errors),
        'full_source_confusions':[{ 'gold':gold,'prediction':pred,'count':int(n)} for (gold,pred),n in errors.groupby(['y_true','y_pred']).size().items()],
        'source_macro_f1':before,
        'oracle_scenarios':{'warning':'Upper-bound accounting with gold labels, not an implemented fix, forecast, or deployable model; no source predictions changed',
            'correct_sample_only':scenario([r['feedback_id'] for r in reviews]),'correct_all_supplied_errors':scenario(errors.feedback_id)},
        'prioritized_fixes':[
            {'priority':1,'fix':'Verify the supplied prediction file and label-ID mapping against its generating model before diagnosing model internals.',
             'reason':'Every one of the 300 supplied errors is parks → roads; the current trained classifier has zero saved validation errors.',
             'expected_delta':'Unknown until provenance is verified. Oracle scenarios quantify the maximum available correction, not a promised gain.'},
            {'priority':2,'fix':'Add paired parks-versus-roads probes covering park walkways and road words inside place names.',
             'reason':'The park-accessibility group and road-named locations are direct candidates for testing context versus keyword shortcuts.',
             'expected_delta':'Unknown; these probes measure whether the hypothesized shortcut exists before any retraining.'},
            {'priority':3,'fix':'Compare clean/noisy versions of park complaints while holding the correct label fixed.',
             'reason':'Hamza variants and elongated prefixes occur in the sample, but clean examples fail as well; normalize only after demonstrating a paired benefit.',
             'expected_delta':'Unknown; secondary spelling features are observations, not established causes.'}],
        'acceptance_note':'A grouped assistant review reduces repetition, but instructor acceptance is needed if it is to replace the specified 120-row human review.'}
    write_json(out/'assistant_review_summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(['Parks → roads'],[120],color='#34699a');axes[0].set_title('Observed confusion');axes[0].set_ylim(0,130)
    names=[DETAILS[k]['title'] for k in GROUPS];counts=[len(GROUPS[k]) for k in GROUPS]
    axes[1].barh(names,counts,color='#438a75');axes[1].set_title('Text groups, not proven root causes')
    for i,n in enumerate(counts):axes[1].text(n+.5,i,str(n),va='center')
    axes[1].set_xlim(0,53);fig.suptitle('Assistant review — 120 examples; human sign-off pending');fig.tight_layout()
    fig.savefig(out/'assistant_error_taxonomy.png',dpi=160);plt.close(fig)
    lines=['# Lab 6 — Short assistant review','',
        '**All 120 examples have been reviewed by the assistant. This is not a claim that you reviewed them.**',
        '', 'You do not need to fill 120 blank explanations to understand the result: the sample contains **45 distinct texts and only three recurring scenarios**, all labelled **parks** and predicted **roads**. Details for every original ID are saved in `artifacts/lab6/assistant_error_review.json`.',
        '', '| Group | Examples | Representative source text | Assistant assessment |','|---|---:|---|---|']
    for key,indices in GROUPS.items():
        row=reviews[indices[0]-1];lines.append(f"| {DETAILS[key]['title']} | {len(indices)} | {row['text']} | {DETAILS[key]['check']} |")
    lines+=['','## The main finding','',
        'Every text explicitly mentions a park (حديقة). The observed category is **topic confusion: parks → roads**, not 120 unrelated defects. The full supplied file has 300 errors, all in this same direction. We cannot tell from the prediction file alone whether the cause is model behaviour, label mapping, or deliberately constructed example predictions.',
        '', 'Thirteen sampled examples contain a hamza variant and/or elongated polite prefix. They remain understandable park requests. Those features are marked per entry, not claimed as causes. Road words in park/place names also deserve paired tests.',
        '', '![Assistant error histogram](../artifacts/lab6/assistant_error_taxonomy.png)',
        '', '## Three proposed fixes','']
    lines += [f"{r['priority']}. **{r['fix']}** {r['reason']} Expected improvement: {r['expected_delta']}" for r in summary['prioritized_fixes']]
    lines+=['','## Quantifying possible correction without inventing results','',
        f"The supplied file's macro-F1 is {before:.4f}. An oracle replacing only the sampled 120 wrong predictions with their gold labels would add {summary['oracle_scenarios']['correct_sample_only']['macro_f1_gain_points']:.2f} macro-F1 points. Correcting all 300 supplied errors would add {summary['oracle_scenarios']['correct_all_supplied_errors']['macro_f1_gain_points']:.2f} points. These are ceilings from label accounting, not measured improvements from a model fix. Nothing was overwritten.",
        '', '## What remains for you','',
        'For understanding, read the three rows above and the spelling caveat. If you want this grouped assistant review accepted instead of the course’s human review, ask the instructor to approve that substitution. Reading three summaries does not automatically certify 120 individual human reviews.',
        '', 'You can send the instructor: “The supplied 120-error sample repeats three parks→roads scenarios. I have an explicitly labelled assistant review for every entry, a grouped summary, and a histogram. May I submit that with a representative human check instead of 120 separate handwritten annotations?”',
        '', 'The original human review fields remain unchanged. No requirement, expected label or source prediction was edited.']
    (ROOT/'docs/LAB6_QUICK_REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ['assistant_reviewed','scenario_histogram','source_macro_f1','oracle_scenarios']},indent=2))


if __name__=='__main__':main()
