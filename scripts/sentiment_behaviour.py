"""Small sentiment baseline required to exercise supplied directional templates."""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from bayan.models.training import ROOT,write_json,sha256
from bayan.preprocessing.core import preprocess


def main():
    frame=pd.read_csv(ROOT/'data/raw/bayan_feedback.csv');train=frame[frame.split=='train'];validation=frame[frame.split=='validation']
    model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True),LinearSVC(random_state=42))
    model.fit(train.text.map(preprocess),train.sentiment)
    labels=model.classes_.tolist(); expected={'negative':-1,'neutral':0,'positive':1}
    if not set(labels).issubset(expected):raise ValueError(f'Unexpected sentiment labels: {labels}')
    rows=pd.read_csv(ROOT/'data/eval/behavioural_templates.csv');rows=rows[rows.test_type=='directional']
    examples=[]
    for row in rows.to_dict('records'):
        negative=row['template'].format(term=row['term']);positive=negative.replace('not working','working').replace('لا تعمل','تعمل')
        examples.append({**row,'before':positive,'after':negative})
    # Ordered labels provide a transparent direction test; no fake probability calibration.
    a=model.predict([preprocess(r['before']) for r in examples]);b=model.predict([preprocess(r['after']) for r in examples])
    for r,before,after in zip(examples,a,b):r.update({'before_prediction':before,'after_prediction':after,'passed':expected[after]<=expected[before]})
    directory=ROOT/'artifacts/sentiment';directory.mkdir(exist_ok=True);joblib.dump(model,directory/'model.joblib')
    report={'model':'TF-IDF (1,2) + LinearSVC, seed 42','scope':'Separate sentiment baseline for directional probes; these are not topic-model direction scores',
        'train_rows':len(train),'validation_rows':len(validation),'data_sha256':sha256(ROOT/'data/raw/bayan_feedback.csv'),
        'validation_macro_f1':float(f1_score(validation.sentiment,model.predict(validation.text.map(preprocess)),average='macro')),
        'passed':sum(r['passed'] for r in examples),'total':len(examples),'checks':examples,'frozen_test_used':False,
        'limitations':'Discrete ordinal sentiment checks allow ties. Passing does not establish sensitivity to negation; report ties separately.',
        'unchanged_predictions':int(np.sum(a==b))}
    write_json(ROOT/'artifacts/lab6/sentiment_directional.json',report);print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
