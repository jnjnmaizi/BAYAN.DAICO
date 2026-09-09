"""Failure-oriented checks for Lab 5–7 implementation boundaries."""
import json
import numpy as np
import pytest
from bayan.evaluation.bootstrap import bootstrap_ci,paired_bootstrap_diff
from bayan.evaluation.slices import sliced_report

@pytest.mark.parametrize('values',[[],[np.nan],[[1,2]]])
def test_bootstrap_rejects_invalid_observations(values):
    with pytest.raises(ValueError):bootstrap_ci(values)

def test_paired_resampling_has_constant_interval_for_constant_difference():
    assert paired_bootstrap_diff([2,3,4],[1,2,3],n_boot=20)==(1.,1.,1.)

def test_absent_gulf_is_unavailable_not_zero_score():
    rows=[{'y_true':'water','y_pred':'water','lang':'ar','dialect_region':'MSA','length_bucket':'short'}]*5
    report=sliced_report(rows,labels=['water','roads'],n_boot=20)
    assert report['dialect_region=Gulf'] is None
    assert report['all']['point']==.5
    assert report['all']['small_slice']

def test_index_corruption_is_rejected_before_model_load(tmp_path):
    from bayan.search.service import CaseSearch
    from bayan.preprocessing.core import PREPROC_VERSION
    prefix=tmp_path/'broken'
    (tmp_path/'broken.faiss').write_bytes(b'corrupt')
    (tmp_path/'broken_manifest.json').write_text(json.dumps({'preproc_version':PREPROC_VERSION,'normalized':True,'index_sha256':'wrong'}))
    with pytest.raises(ValueError,match='checksum'):CaseSearch(str(prefix))

def test_empty_search_does_not_invent_results():
    from bayan.search.service import CaseSearch
    search=CaseSearch.__new__(CaseSearch)
    assert search.search('  ')==[]

def test_api_rejects_missing_empty_and_oversized_text():
    from fastapi.testclient import TestClient
    from bayan.serving.api import app
    client=TestClient(app)
    for body in [{},{'text':''},{'text':'x'*10001}]:
        assert client.post('/v1/classify',json=body).status_code==422

def test_microbatch_keeps_request_order_and_runs_one_inference():
    import asyncio
    from bayan.serving.batching import MicroBatcher
    class FakeRuntime:
        calls=0
        def tokenize(self,texts):return texts
        def logits(self,texts):
            self.calls+=1
            return np.array([[len(text),0] for text in texts])
    async def check():
        model=FakeRuntime();batcher=MicroBatcher(model)
        results=await asyncio.gather(batcher.submit('a'),batcher.submit('abcd'))
        assert [int(r[0]) for r in results]==[1,4]
        assert model.calls==1
        batcher.worker.cancel()
        try:await batcher.worker
        except asyncio.CancelledError:pass
    asyncio.run(check())

def test_search_and_combined_analysis_use_same_dedicated_worker(monkeypatch):
    import threading
    from fastapi.testclient import TestClient
    import bayan.serving.api as api
    called=[]
    class FakeSearch:
        def search(self,*args,**kwargs):
            called.append(threading.get_ident());return []
    async def fake_classify(payload):return {'topic':'water'}
    monkeypatch.setattr(api,'ensure_ready',lambda:None)
    monkeypatch.setattr(api,'search_service',lambda:FakeSearch())
    monkeypatch.setattr(api,'classify',fake_classify)
    monkeypatch.setattr(api,'entities',lambda payload:{'entities':[]})
    client=TestClient(api.app)
    assert client.post('/v1/search',json={'text':'water'}).status_code==200
    assert client.post('/v1/analyse',json={'text':'water'}).status_code==200
    assert len(called)==2 and called[0]==called[1]
    assert called[0]!=threading.get_ident()

def test_entity_path_keeps_reference_digits_and_source_offsets(monkeypatch):
    import bayan.serving.api as api
    class Batch:
        def word_ids(self,index):return [None,0,None]
    class Model:
        labels={0:'B-REFERENCE'}
        def tokenize(self,texts,words):
            assert texts==[['BYN-2026-000001']]
            return Batch()
        def logits(self,batch):return np.zeros((1,3,1))
    monkeypatch.setattr(api,'ensure_ready',lambda:None)
    monkeypatch.setattr(api,'runtime',lambda task:Model())
    result=api.entities(api.Request(text='BYN-2026-000001'))
    assert result['entities'][0]['text']=='BYN-2026-000001'
    assert result['entities'][0]['start']==0
    assert result['entities'][0]['end']==len('BYN-2026-000001')
