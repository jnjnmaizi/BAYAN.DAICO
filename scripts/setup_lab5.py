"""Fetch only the pinned retrieval model files needed by Lab 5."""
from huggingface_hub import snapshot_download
from bayan.search.index import sources
from bayan.models.training import ROOT

for repo,source in sources().items():
    snapshot_download(repo,revision=source['revision'],cache_dir=ROOT/'artifacts/hf_cache',
        allow_patterns=['*.json','*.safetensors','sentencepiece.bpe.model','vocab.txt','*.model'],
        ignore_patterns=['onnx/*','openvino/*'])
    print(repo,source['revision'])
