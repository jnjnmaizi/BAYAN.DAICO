"""Choose the fastest measured backend whose paired quality tax CI is acceptable."""
import json
from bayan.models.training import ROOT,write_json
from bayan.preprocessing.core import PREPROC_VERSION


def main():
    directory=ROOT/'artifacts/lab7';result={'preproc_version':PREPROC_VERSION,'policy':'Fastest measured dynamic-128 p99 among candidates with quality-tax upper 95% bound <= 0.01; keep original fp32 rollback'}
    for task in ['classifier','ner']:
        baseline=json.loads((directory/f'{task}_torch_benchmark.json').read_text())
        choices=[{'backend':'torch','p99_ms':baseline['rows']['dynamic_128']['p99_ms'],'quality_tax_high':0.}]
        for backend in ['onnx','int8']:
            timing=json.loads((directory/f'{task}_{backend}_benchmark.json').read_text());quality=json.loads((directory/f'{task}_{backend}_quality.json').read_text())
            high=quality['quality_tax_baseline_minus_candidate']['high']
            if high<=.01:choices.append({'backend':backend,'p99_ms':timing['rows']['dynamic_128']['p99_ms'],'quality_tax_high':high})
        chosen=min(choices,key=lambda row:row['p99_ms']);chosen['speedup_vs_padded_p99']=baseline['rows']['padded_512']['p99_ms']/chosen['p99_ms']
        chosen['bare_p99_target_met']=chosen['p99_ms']<=25;chosen['speedup_target_met']=chosen['speedup_vs_padded_p99']>=6;result[task]=chosen
    write_json(directory/'selection.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
