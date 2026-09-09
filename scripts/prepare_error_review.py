"""Prepare the mandated human review without automatically assigning categories."""
import json
import pandas as pd
from bayan.models.training import ROOT, write_json, sha256


def main():
    predictions=ROOT/'data/eval/validation_predictions.csv'
    source=pd.read_csv(predictions)
    raw=pd.read_csv(ROOT/'data/raw/bayan_feedback.csv')
    errors=source[source.y_true!=source.y_pred].sample(n=120,random_state=42).merge(raw[['feedback_id','text']],on='feedback_id',validate='one_to_one')
    destination=ROOT/'artifacts/lab6';destination.mkdir(exist_ok=True)
    review=destination/'human_error_review.json'
    if review.exists(): raise SystemExit('Review exists; preserve human annotations')
    rows=[{**r,'category':None,'reviewer_note':None,'human_confirmed':False} for r in errors.to_dict('records')]
    write_json(review,rows)
    write_json(destination/'review_protocol.json',{'source':str(predictions.relative_to(ROOT)),'sha256':sha256(predictions),'source_error_count':int((source.y_true!=source.y_pred).sum()),'sample_size':120,'seed':42,'source_is':'supplied course predictions, not current trained classifier predictions','human_review_complete':False})
    lines=['# Lab 6 — Human error review','', 'These 120 errors come from the supplied course predictions, not our trained classifier. Read each text with its gold and predicted topic. Add a category and note, then confirm it in `artifacts/lab6/human_error_review.json`. The report counts only explicitly confirmed entries.','', 'Categories: label ambiguity; Arabic spelling; dialect/code-switching; entity alignment; truncation; retrieval relevance; preprocessing/serving skew; annotation defect; unexplained model confusion. Do not infer a root cause merely from the wrong label.','']
    for i,r in enumerate(rows,1):
        lines += [f"## {i}. {r['feedback_id']}",'',r['text'],'',f"Gold: **{r['y_true']}** · Prediction: **{r['y_pred']}**",'', '- Category:','- Evidence / note:','- Human confirmed: no','']
    (ROOT/'docs/LAB6_ERROR_REVIEW.md').write_text('\n'.join(lines))


if __name__=='__main__': main()
