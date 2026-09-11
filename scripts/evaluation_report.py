"""Generate measured slices and model cards, preserving the human-review gate."""
import argparse
import hashlib
import json
import pandas as pd
import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from bayan.models.training import ROOT,write_json
from bayan.preprocessing.core import preprocess,PREPROC_VERSION
from bayan.preprocessing.arabic import prepare_arabic_text,CAMELBERT_V1
from bayan.evaluation.slices import sliced_report,f1_interval
from bayan.evaluation.behavioural import run_behavioural_suite


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(4)
    out=ROOT/'artifacts/lab6';out.mkdir(exist_ok=True)
    raw=pd.read_csv(ROOT/'data/raw/bayan_feedback.csv')
    labels=sorted(raw.topic.unique())
    frames={}
    frames['topic']=pd.DataFrame(json.loads((ROOT/'artifacts/topic_classifier/validation_predictions.json').read_text())).rename(columns={'label':'y_true','prediction':'y_pred'})
    frames['dialect_aware']=pd.DataFrame(json.loads((ROOT/'artifacts/lab4_models/da/validation_predictions.json').read_text())).rename(columns={'gold':'y_true','prediction':'y_pred'})
    report={}
    for name,frame in frames.items():
        frame=frame[['feedback_id','y_true','y_pred']].merge(raw[['feedback_id','citizen_group_id','lang','dialect_region','text']],on='feedback_id',validate='one_to_one')
        frame['length_bucket']=frame.text.str.split().str.len().map(lambda n:'short' if n<10 else 'medium' if n<25 else 'long')
        frames[name]=frame
        evidence=out/f'{name}.json'
        if evidence.exists(): report[name]=json.loads(evidence.read_text());continue
        directory=ROOT/('artifacts/topic_classifier' if name=='topic' else 'artifacts/lab4_models/da')
        tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(directory,local_files_only=True).to(args.device).eval()
        def predict(texts):
            inputs=[preprocess(t) for t in texts]
            if name=='dialect_aware': inputs=[prepare_arabic_text(t,CAMELBERT_V1)['model_text'] for t in inputs]
            result=[]
            with torch.inference_mode():
                for start in range(0,len(inputs),32):
                    batch=tokenizer(inputs[start:start+32],padding=True,truncation=True,max_length=64,return_tensors='pt').to(args.device)
                    result.extend(model.config.id2label[int(i)] for i in model(**batch).logits.argmax(-1).cpu().tolist())
            return result
        behaviour=run_behavioural_suite(predict)
        report[name]={'slices':sliced_report(frame,labels),'behavioural':behaviour,'scope':'saved validation predictions for slices; new development probes for behaviour'}
        write_json(evidence,report[name]);del model
    joined=frames['topic'].merge(frames['dialect_aware'][['feedback_id','y_pred']],on='feedback_id',suffixes=('','_da'),validate='one_to_one')
    paired=f1_interval(joined,labels,other=joined.y_pred_da)
    review=json.loads((out/'human_error_review.json').read_text())
    confirmed=[r for r in review if r.get('human_confirmed') is True and r.get('category') and r.get('reviewer_note')]
    human_complete=len(review)==120 and len(confirmed)==120
    human_status='complete (120/120, confirmed by the reviewer)' if human_complete else f'pending ({len(confirmed)}/120 confirmed)'
    histogram=pd.Series([r['category'] for r in confirmed],dtype=str).value_counts().to_dict()
    if histogram:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axis=plt.subplots(figsize=(9,5))
        axis.barh(list(histogram),list(histogram.values()))
        axis.set_xlabel('Human-confirmed errors');axis.set_title(f'Error taxonomy: {len(confirmed)}/120 reviewed')
        fig.tight_layout();fig.savefig(out/'error_taxonomy.png',dpi=160);plt.close(fig)
    else:
        # A withdrawn review must not leave a stale completed-review chart.
        (out/'error_taxonomy.png').unlink(missing_ok=True)

    summary={'paired_topic_minus_da_on_same_arabic_rows':paired,'human_review':{'confirmed':len(confirmed),'required':120,'complete':human_complete,'histogram':histogram,
        'methods':sorted({r.get('review_method','individual confirmation') for r in confirmed})},
        'slice_counts':{k:len(v['slices']) for k,v in report.items()},'frozen_test_used':False,
        'top_3_fixes':[{'fix':'Add independent Gulf and complete-class evaluation coverage','predicted_delta':'Not estimable from current data; improves coverage, not an asserted model gain'},
                       {'fix':'Improve sparse case-ID relevance judgements before retrieval tuning','predicted_delta':'Unknown pending human relevance review'},
                       {'fix':'Train with consistent clitic segmentation when deploying segmented NER','predicted_delta':'Measured recovery +23.6364 recall points versus mismatched input; 0 versus original baseline'}]}
    assistant_path=out/'assistant_review_summary.json'
    assistant=json.loads(assistant_path.read_text()) if assistant_path.exists() else None
    if assistant:
        summary['assistant_review']={'reviewed':assistant['assistant_reviewed'],
            'scenario_histogram':assistant['scenario_histogram'],
            'primary_category_histogram':assistant['primary_category_histogram'],
            'counts_as_human_review':False}
        summary['top_3_fixes']=[{'fix':r['fix'],'predicted_delta':r['expected_delta']} for r in assistant['prioritized_fixes']]
    directional_path=out/'sentiment_directional.json'
    directional=json.loads(directional_path.read_text()) if directional_path.exists() else None
    summary['sentiment_directional']={k:directional[k] for k in ['passed','total','unchanged_predictions','validation_macro_f1']} if directional else None
    grouped_path=out/'human_group_review.json'
    grouped=json.loads(grouped_path.read_text()) if grouped_path.exists() else None
    if grouped:
        submitted=ROOT/grouped['submitted_file']
        if hashlib.sha256(submitted.read_bytes()).hexdigest()!=grouped['submitted_file_sha256']:
            raise ValueError('Submitted grouped review changed; validate it again before reporting')
        by_id={r['feedback_id']:r for r in review}
        covered=[]
        for group in grouped['groups']:
            if not group['reviewer_notes'] or group['decision'] not in ['Agree','Disagree','Unsure']:
                raise ValueError('Grouped review requires explicit decisions and notes')
            for fid in group['feedback_ids']:
                if fid not in by_id or any(by_id[fid][k]!=group[k] for k in ['text','y_true','y_pred']):
                    raise ValueError('Grouped review no longer matches source entries')
                covered.append(fid)
        if len(covered)!=len(set(covered)):
            raise ValueError('Grouped review repeats source IDs')
        summary['human_group_review']={'groups_reviewed':len(grouped['groups']),
            'source_entries_covered':len(covered),'decisions':grouped['decisions'],
            'individual_human_confirmations_added':0,'grouped_summary_only':True}
    alternatives_path=ROOT/'artifacts/alternative_evidence.json'
    alternatives=json.loads(alternatives_path.read_text()) if alternatives_path.exists() else None
    if alternatives:
        if alternatives.get('status')!='documented_additional_evidence':
            raise ValueError('Alternative evidence record has an unexpected status')
        summary['alternative_evidence']={
            'scope': alternatives['scope'],
            'status': alternatives['status'],
            'raw_measurements_preserved': alternatives['raw_measurements_preserved'],
            'course_requirements_changed': alternatives['course_requirements_changed']
        }
    write_json(out/'summary.json',summary)
    acceptance_line=('Labs 3–5 have additional evidence documented; their raw metrics and original course requirements remain unchanged. See [the evidence record](artifacts/alternative_evidence.json).' if alternatives else '')
    lines=['# Evaluation report — Jana Alhumaizi NLP','', 'Validation performance reaches the ceiling of this synthetic dataset, while the retrieval system misses the supplied exact-ID targets. Gulf generalisation remains unmeasurable. Human error review is '+human_status+'.', acceptance_line,'', '## Sliced metrics with 95% bootstrap intervals','', '500 seeded bootstrap draws; citizen groups are resampled together. All eight topic labels stay in macro-F1, including absent labels. Slices with fewer than 30 groups are flagged. These are validation estimates, not new frozen-test results.','', '| Model / slice | Rows | Macro-F1 [95% CI] | Small slice |','|---|---:|---|---|']
    for name,data in report.items():
        for key,value in data['slices'].items():
            lines.append(f"| {name} / {key} | {value['n'] if value else 0} | "+(f"{value['point']:.4f} [{value['low']:.4f}, {value['high']:.4f}] | {value['small_slice']} |" if value else 'Unavailable | N/A |'))
    lines+=['','Paired topic minus DA macro-F1 on the same Arabic rows: '+json.dumps(paired)+'. No Gulf rows are present.','', '## Behavioural suite','', '| Model | Invariance | MFT | Directional |','|---|---|---|---|']
    for name,data in report.items():
        b=data['behavioural'];lines.append(f"| {name} | {b['invariance']['passed']}/{b['invariance']['total']} | {b['mft']['passed']}/{b['mft']['total']} | Not assessable: topic model has no sentiment output |")
    if directional:
        lines += ['', f"Separate TF-IDF sentiment baseline: {directional['passed']}/{directional['total']} directional checks passed, with {directional['unchanged_predictions']} unchanged predictions. Validation sentiment macro-F1={directional['validation_macro_f1']:.4f}. The test permits ties; this result does not demonstrate sensitivity to negation. Training uses only the supplied training split."]
    lines+=['','Invariance uses the supplied 200 templates with whitespace perturbations. Sixteen explicit bilingual topic probes supplement the supplied file, which has no MFT rows. The 200 directional templates require sentiment output; topic probabilities cannot substitute for sentiment. DA English probes are deliberately reported as an out-of-scope stress test.','', '## Error taxonomy','', f"Human-confirmed review: **{len(confirmed)}/120**. The [individual worksheet](docs/LAB6_ERROR_REVIEW.md) and `artifacts/lab6/human_error_review.json` retain per-entry decisions separately. The sample uses the course-provided predictions, which contain 300 errors; our saved topic validation predictions contain no errors. These two sources are not interchangeable.", '', 'Individual-entry category counts: '+json.dumps(histogram)+'. Grouped decisions are reported above.','', '### Top three proposed fixes (not measured promises)','']
    if assistant:
        # Keep assistant observations separate from the human confirmation count.
        position=lines.index('### Top three proposed fixes (not measured promises)')
        lines[position:position]=[
            'Draft annotation coverage: **'+str(assistant['assistant_reviewed'])+'/120** entries. The sample contains three scenarios: playground maintenance (42), park accessibility (46), and park irrigation (32). All have the observed confusion parks → roads. These scenarios do not establish a model-internal cause.',
            '', 'See the [review summary](docs/LAB6_QUICK_REVIEW.md) and [review method and provenance](docs/LAB6_REVIEW_METHOD.md). Draft annotations and submitted decisions remain separate records.',
            '', ('![Individual-review histogram](artifacts/lab6/error_taxonomy.png)' if human_complete else ''), '',
            f"Using gold labels only as an accounting exercise, correcting the 120 sampled predictions would add {assistant['oracle_scenarios']['correct_sample_only']['macro_f1_gain_points']:.2f} macro-F1 points; correcting all 300 supplied errors would add {assistant['oracle_scenarios']['correct_all_supplied_errors']['macro_f1_gain_points']:.2f}. These are correction ceilings, not trained-model gains. Source predictions remain unchanged.", '']
    lines += [f"- {r['fix'].rstrip('.')}: {r['predicted_delta'].rstrip('.')}." for r in summary['top_3_fixes']]
    retrieval=json.loads((ROOT/'artifacts/lab5/retrieval.json').read_text())
    lines+=['','## Retrieval quality','',f"Reranked recall@10={retrieval['reranked']['recall_at_10']:.4f}; MRR@10={retrieval['reranked']['mrr_at_10']:.4f}. Empty-correct={retrieval['no_answer_empty_correct']}/20 on the calibration queries, which contain only one unique no-answer text. This is not independent rejection accuracy. See `artifacts/lab5/retrieval.json`.",'','## Known limitations','', '- All source data are synthetic; repeated templates limit generalisation claims.','- No Gulf validation rows, no Arabic topic test rows, and only four NER validation template groups.','- Exact-ID retrieval labels are sparse among 20,000 repeated cases; original labels remain unchanged.','- Human review is not complete; directional probes pass through ties on a weak separate sentiment baseline. The Lab 6 targets must not be marked fully achieved.','- Confidence intervals describe this dataset and resampling protocol; they do not repair missing dialect or entity coverage.','']
    lines=[line.replace('- Human review is not complete; directional probes pass through ties on a weak separate sentiment baseline. The Lab 6 targets must not be marked fully achieved.', '- Human review is '+human_status+'. Directional probes pass through ties on a weak separate sentiment baseline; they do not establish negation understanding.') for line in lines]
    if alternatives:
        position=lines.index('## Known limitations')
        title='## Additional evidence for Labs 3–5'
        text='Labs 3–5 have additional evidence documented. The feasibility audit, paired NER comparison, retrieval integrity/relevance audit and hybrid development probe are retained as supporting analysis. Original exact-ID scores, labels, datasets and course requirements remain unchanged. See [the evidence record](artifacts/alternative_evidence.json).'
        lines[position:position]=[title,'',text,'']
    if grouped:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        group_counts={decision:sum(g['decision']==decision for g in grouped['groups']) for decision in ['Agree','Disagree','Unsure']}
        row_counts={decision:sum(len(g['feedback_ids']) for g in grouped['groups'] if g['decision']==decision) for decision in group_counts}
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for axis,counts,title in zip(axes,[group_counts,row_counts],['Submitted group decisions','Source entries represented']):
            axis.bar(list(counts),list(counts.values()),color=['#0d9488','#f59e0b','#94a3b8'])
            axis.set_title(title);axis.set_ylim(0,max(counts.values())*1.2 or 1)
            for i,value in enumerate(counts.values()):axis.text(i,value,str(value),ha='center',va='bottom')
        fig.suptitle('Lab 6 — Grouped review results')
        fig.tight_layout();fig.savefig(out/'grouped_review_histogram.png',dpi=160);plt.close(fig)
        position=lines.index('## Error taxonomy')+1
        acceptance='The grouped review is retained as a compact summary; the individual review is counted separately.'
        lines[position:position]=['', f"**Grouped human review received: {len(grouped['groups'])}/45 groups**, covering {len(covered)}/120 original entries. Each group includes a submitted decision and the reviewer’s notes. See the [completed grouped worksheet](docs/LAB6_SHORT_REVIEW.md). Grouped decisions are recorded separately from per-entry confirmations. "+acceptance,'','![Grouped review results](artifacts/lab6/grouped_review_histogram.png)','']
    lines=[line.replace('Human-confirmed review: **', 'Individual-entry confirmations (separate from grouped decisions): **') for line in lines]
    body='\n'.join(lines)
    while '\n\n\n' in body:
        body=body.replace('\n\n\n','\n\n')
    (ROOT/'EVALUATION_REPORT.md').write_text(body.rstrip()+'\n')
    cards=ROOT/'model_cards';cards.mkdir(exist_ok=True)
    for name,checkpoint,limitations in [('topic','xlm-roberta-base','Synthetic repeated feedback; no Gulf validation and no Arabic frozen test. Eight-label macro-F1 can be capped by missing classes.'),('dialect_aware','CAMeL-Lab/bert-base-arabic-camelbert-da','Evaluated only on MSA; no evidence of Gulf improvement. English behaviour is outside intended Arabic use.'),('ner','xlm-roberta-base','Only four validation templates and two frozen-test templates; no ORGANISATION labels and fixed dates. High entity-F1 does not imply real-world entity coverage.')]:
        metrics=report[name]['slices']['all'] if name in report else json.loads((ROOT/'artifacts/lab3/ner.json').read_text())['validation']
        template=Environment(loader=FileSystemLoader(ROOT/'templates'),undefined=StrictUndefined,autoescape=False).get_template('model_card.md.j2')
        training_path=ROOT/('artifacts/lab3/ner.json' if name=='ner' else 'artifacts/lab3/topic_classifier.json' if name=='topic' else 'artifacts/lab4/camelbert_da.json')
        training=json.loads(training_path.read_text())
        behaviour=report[name]['behavioural'] if name in report else None
        body=template.render(model_name=f'Jana Alhumaizi NLP — {name}',checkpoint=checkpoint+' @ '+training['revision'],
            intended_use=('Entity extraction' if name=='ner' else 'Citizen-feedback topic classification')+' for course demonstration; human review required for operational decisions.',
            preproc_version=PREPROC_VERSION,data_version=training.get('data_sha256',training.get('protocol',{}).get('data_sha256','See pinned training report')),
            metrics_table='```json\n'+json.dumps(metrics,indent=2)+'\n```',
            slices_table='See [evaluation report](../EVALUATION_REPORT.md) and the corresponding Lab 3/4/6 evidence. Missing dialects and entity types are not assigned zero performance.',
            behavioural_table=(f"Invariance {behaviour['invariance']['passed']}/{behaviour['invariance']['total']}; bilingual MFT {behaviour['mft']['passed']}/{behaviour['mft']['total']}. Sentiment direction uses a separate baseline." if behaviour else 'Classification probes do not apply to NER. Label-alignment contracts and paired quantization validation are recorded in the test suite and Lab 7 evidence.'),
            known_limitations=limitations,owner='Repository maintainer; educational project.')+'\n'
        (cards/f'{name}.md').write_text(body)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
