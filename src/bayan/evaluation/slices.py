"""Macro-F1 slices with paired, optionally group-level bootstrap resampling."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def f1_interval(frame, labels, n_boot=500, seed=42, other=None):
    if frame.empty:
        return None
    # Resample citizen groups together where available, otherwise individual rows.
    keys=frame['citizen_group_id'].astype(str).to_numpy() if 'citizen_group_id' in frame else np.arange(len(frame))
    unique=np.unique(keys)
    groups=[np.flatnonzero(keys==key) for key in unique]
    gold=frame.y_true.to_numpy(); pred=frame.y_pred.to_numpy()
    def score(ix):
        value=f1_score(gold[ix],pred[ix],labels=labels,average='macro',zero_division=0)
        if other is not None:
            value-=f1_score(gold[ix],np.asarray(other)[ix],labels=labels,average='macro',zero_division=0)
        return value
    rng=np.random.default_rng(seed)
    samples=[score(np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])) for _ in range(n_boot)]
    return {'point':float(score(np.arange(len(frame)))),'low':float(np.quantile(samples,.025)),
            'high':float(np.quantile(samples,.975)),'n':len(frame),'groups':len(groups),'small_slice':len(groups)<30}


def sliced_report(frame, labels=None, n_boot=500):
    frame=pd.DataFrame(frame).copy()
    labels=labels or sorted(frame.y_true.unique().tolist())
    output={'all':f1_interval(frame,labels,n_boot)}
    for column in ['lang','dialect_region','y_true','length_bucket']:
        for value in sorted(frame[column].fillna('unknown').unique()):
            subset=frame[frame[column].fillna('unknown')==value]
            output[f'{column}={value}']=f1_interval(subset,labels,n_boot)
    for dialect in ['Gulf','MSA']:
        output.setdefault(f'dialect_region={dialect}',None)
    return output
