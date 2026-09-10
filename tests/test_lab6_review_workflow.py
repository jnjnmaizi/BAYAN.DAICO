"""The review interface records explicit human decisions only."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('review_lab6',ROOT/'scripts/review_lab6.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(tmp_path):
    row={'feedback_id':'FB-1','text':'ألعاب الأطفال في حديقة جدة تحتاج صيانة','y_true':'parks','y_pred':'roads',
         'category':None,'reviewer_note':None,'human_confirmed':False}
    path=tmp_path/'review.json';path.write_text(json.dumps([row]))
    suggestion={**row,'assistant_category':'Topic confusion','assistant_note':'The park equipment needs maintenance.'}
    return path,{'FB-1':suggestion}


def test_quitting_or_skipping_does_not_confirm_anything(tmp_path):
    for action in ['q','s','']:
        path,suggestions=fixture(tmp_path);before=path.read_bytes()
        assert module.review_entries(path,suggestions,input_fn=lambda _:action,output_fn=lambda _:None)==0
        assert path.read_bytes()==before


def test_explicit_adoption_is_saved_and_not_asked_again(tmp_path):
    path,suggestions=fixture(tmp_path)
    assert module.review_entries(path,suggestions,input_fn=lambda _:'a',output_fn=lambda _:None)==1
    row=json.loads(path.read_text())[0]
    assert module.is_confirmed(row)
    assert row['review_method'].startswith('human explicitly adopted')
    def no_more_prompts(_):raise AssertionError('Confirmed entry was repeated')
    assert module.review_entries(path,suggestions,input_fn=no_more_prompts,output_fn=lambda _:None)==0


def test_custom_assessment_is_preserved(tmp_path):
    path,suggestions=fixture(tmp_path);answers=iter(['e','Label ambiguity','I need the service ownership definition.'])
    module.review_entries(path,suggestions,input_fn=lambda _:next(answers),output_fn=lambda _:None)
    row=json.loads(path.read_text())[0]
    assert row['category']=='Label ambiguity'
    assert row['reviewer_note']=='I need the service ownership definition.'


def test_stale_suggestion_cannot_be_confirmed(tmp_path):
    path,suggestions=fixture(tmp_path);suggestions['FB-1']['text']='Unrelated text';before=path.read_bytes()
    assert module.review_entries(path,suggestions,input_fn=lambda _:'a',output_fn=lambda _:None)==0
    assert path.read_bytes()==before
