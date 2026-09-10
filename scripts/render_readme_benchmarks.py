"""Render README performance evidence from saved runs; no model inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    colors = ['#CBD5E1', '#94A3B8', '#38BDF8', '#0D9488']
    labels = ['PyTorch / padded 512', 'PyTorch / dynamic 128',
              'ONNX FP32 / dynamic 128', 'ONNX INT8 / dynamic 128']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharex=True)
    fig.patch.set_facecolor('#F8FAFC')
    fig.suptitle('Faster CPU inference, measured on the same inputs',
                 x=.04, y=.96, ha='left', fontsize=19, weight='bold', color='#0F172A')
    for ax, task, title in zip(axes, ['classifier', 'ner'], ['Topic classification', 'Entity extraction']):
        reports = {backend: json.loads((ROOT / f'artifacts/lab7/{task}_{backend}_benchmark.json').read_text())
                   for backend in ['torch', 'onnx', 'int8']}
        assert reports['torch']['sample_indices'] == reports['onnx']['sample_indices'] == reports['int8']['sample_indices']
        values = [reports[backend]['rows'][mode]['p99_ms'] for backend, mode in
                  [('torch', 'padded_512'), ('torch', 'dynamic_128'), ('onnx', 'dynamic_128'), ('int8', 'dynamic_128')]]
        ax.set_facecolor('#F8FAFC')
        ax.barh(range(4), values, color=colors, height=.55, zorder=3)
        ax.set_yticks(range(4), labels, fontsize=10)
        ax.invert_yaxis()
        for i, value in enumerate(values):
            ax.text(value + 3, i, f'{value:.2f} ms', va='center', fontsize=10, color='#0F172A')
        ax.set_title(title, loc='left', pad=16, fontsize=13, weight='bold', color='#0F172A')
        ax.set_xlabel('p99 model latency (ms) · lower is better', fontsize=10, labelpad=12)
        ax.set_xlim(0, 230)
        ax.xaxis.grid(True, color='#E2E8F0', zorder=0)
        ax.tick_params(length=0, colors='#475569')
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.04, .05, 'Apple Silicon CPU · 4 threads · batch 1 · 200 sampled inputs · 10 warm-ups\n'
             'Model execution only; excludes tokenization and HTTP. Baseline pads to 512; dynamic runs cap at 128.',
             fontsize=10, color='#475569', linespacing=1.6)
    fig.subplots_adjust(left=.18, right=.97, top=.79, bottom=.25, wspace=.95)
    output = ROOT / 'docs/assets/inference_latency.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
