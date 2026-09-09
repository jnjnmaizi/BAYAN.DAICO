"""Refuse incompatible selected artifacts before accepting requests."""
import json
from bayan.models.training import ROOT,sha256
from bayan.preprocessing.core import PREPROC_VERSION,preprocess


def run_startup_canaries(runtime=None):
    from bayan.serving.runtime import Runtime
    selection=json.loads((ROOT/'artifacts/lab7/selection.json').read_text())
    runtime=runtime or Runtime('classifier',selection['classifier']['backend'])
    if selection['preproc_version']!=PREPROC_VERSION: raise ValueError('Preprocessing version skew')
    assert preprocess('  0551234567   test  ')=='<PHONE> test'
    for task in ['classifier','ner']:
        selected=selection[task]
        if selected['backend']!='torch':
            directory=ROOT/'artifacts/onnx'/task
            manifest=json.loads((directory/'manifest.json').read_text())
            source=ROOT/'artifacts'/('topic_classifier' if task=='classifier' else 'ner')
            if sha256(source/'tokenizer.json')!=manifest['tokenizer_sha256'] or sha256(source/'config.json')!=manifest['config_sha256']:
                raise ValueError('Tokenizer or label configuration skew')
            kind='int8' if selected['backend']=='int8' else 'fp32'
            if manifest['preproc_version']!=PREPROC_VERSION or sha256(directory/f'{kind}.onnx')!=manifest[f'{kind}_sha256']:
                raise ValueError('Selected model checksum/version mismatch')
    texts=['I paid the bill but it still shows as unpaid','الحاوية ممتلئة ولم يتم جمع النفايات']
    expected=['billing','waste'];actual=runtime.predict(texts)
    if actual!=expected: raise ValueError(f'Classifier canary failed: {actual}')
    return {'passed':True,'preproc_version':PREPROC_VERSION,'classifier_predictions':actual,'checks':['selected artifact checksums','preprocessing version','PII normalization','English billing','Arabic waste']}
