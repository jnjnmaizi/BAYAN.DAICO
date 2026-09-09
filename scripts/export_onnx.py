"""Export measured Lab 3 weights to ONNX and dynamic MatMul INT8 on CPU."""
import argparse
import json
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
from bayan.serving.runtime import Runtime
from bayan.models.training import ROOT,write_json,sha256
from bayan.preprocessing.core import PREPROC_VERSION


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['classifier','ner'],default='classifier');args=parser.parse_args()
    if not (ROOT/'artifacts/lab7'/f'{args.task}_torch_benchmark.json').exists():
        raise SystemExit('Measure the original PyTorch baseline before export')
    directory=ROOT/'artifacts/onnx'/args.task;directory.mkdir(parents=True,exist_ok=True)
    fp=directory/'fp32.onnx';q=directory/'int8.onnx'
    if (directory/'manifest.json').exists(): print('Existing export preserved');return
    runtime=Runtime(args.task)
    class Logits(torch.nn.Module):
        def __init__(self,model):super().__init__();self.model=model
        def forward(self,input_ids,attention_mask):return self.model(input_ids=input_ids,attention_mask=attention_mask).logits
    batch=runtime.tokenize(['The road needs repair.','الشارع يحتاج صيانة'])
    if not fp.exists():
        print('Exporting',args.task,flush=True)
        torch.onnx.export(Logits(runtime.model),(batch['input_ids'],batch['attention_mask']),str(fp),
            input_names=['input_ids','attention_mask'],output_names=['logits'],
            dynamic_axes={'input_ids':{0:'batch',1:'sequence'},'attention_mask':{0:'batch',1:'sequence'},'logits':{0:'batch',**({1:'sequence'} if args.task=='ner' else {})}},
            opset_version=17,dynamo=False)
    del runtime
    print('Quantizing MatMul weights to signed INT8',flush=True)
    if not q.exists(): quantize_dynamic(str(fp),str(q),weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul'],per_channel=True,extra_options={'MatMulConstBOnly':True})
    write_json(directory/'manifest.json',{'task':args.task,'preproc_version':PREPROC_VERSION,
        'source_weights_sha256':sha256(ROOT/'artifacts'/('topic_classifier' if args.task=='classifier' else 'ner')/'model.safetensors'),
        'fp32_sha256':sha256(fp),'int8_sha256':sha256(q),'opset':17,'quantization':'dynamic signed INT8 per-channel MatMul constant weights; embeddings remain fp32','rollback':'fp32.onnx',
        'tokenizer_sha256':sha256(ROOT/'artifacts'/('topic_classifier' if args.task=='classifier' else 'ner')/'tokenizer.json'),
        'config_sha256':sha256(ROOT/'artifacts'/('topic_classifier' if args.task=='classifier' else 'ner')/'config.json'),
        'quality_status':'Must pass paired validation before selection'})
    print('Export complete',flush=True)


if __name__=='__main__': main()
