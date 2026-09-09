"""Shared CPU inference used by benchmarks and HTTP serving."""
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from bayan.models.training import ROOT
from bayan.preprocessing.core import preprocess


class Runtime:
    def __init__(self, task='classifier', backend='torch', threads=4):
        self.task=task; self.backend=backend
        self.directory=ROOT/'artifacts'/('topic_classifier' if task=='classifier' else 'ner')
        self.config=json.loads((self.directory/'config.json').read_text())
        self.tokenizer=AutoTokenizer.from_pretrained(self.directory,local_files_only=True)
        self.labels={int(k):v for k,v in self.config['id2label'].items()}
        torch.set_num_threads(threads)
        if backend=='torch':
            cls=AutoModelForSequenceClassification if task=='classifier' else AutoModelForTokenClassification
            self.model=cls.from_pretrained(self.directory,local_files_only=True,attn_implementation='eager').eval()
        else:
            import onnxruntime as ort
            options=ort.SessionOptions(); options.intra_op_num_threads=threads; options.inter_op_num_threads=1
            options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.path=ROOT/'artifacts/onnx'/task/('int8.onnx' if backend=='int8' else 'fp32.onnx')
            self.session=ort.InferenceSession(str(self.path),sess_options=options,providers=['CPUExecutionProvider'])

    def tokenize(self,texts,max_length=128,padding=True,words=False):
        return self.tokenizer(texts if words else [preprocess(t) for t in texts],is_split_into_words=words,
            padding=padding,truncation=True,max_length=max_length,return_tensors='pt' if self.backend=='torch' else 'np')

    def logits(self,batch):
        if self.backend=='torch':
            with torch.inference_mode(): return self.model(**batch).logits.numpy()
        names={v.name for v in self.session.get_inputs()}
        return self.session.run(['logits'],{k:np.asarray(v,dtype=np.int64) for k,v in batch.items() if k in names})[0]

    def predict(self,texts,batch_size=32,max_length=128):
        outputs=[]
        for start in range(0,len(texts),batch_size):
            logits=self.logits(self.tokenize(texts[start:start+batch_size],max_length))
            outputs.extend([self.labels[int(i)] for i in logits.argmax(-1)])
        return outputs
