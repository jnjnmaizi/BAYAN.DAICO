"""Additional leakage, entity-boundary and QA regressions; no model downloads."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bayan.models.data import build_topic_dataset
from bayan.models.ner import align_labels, read_conll, split_ner
from bayan.models.qa import best_span
from bayan.models.training import topic_metrics, write_json


def topic_csv(tmp_path, change=None):
    rows = [{"feedback_id":f"F{i}", "citizen_group_id":f"C{i}",
             "text":"الخدمــــة   جيدة", "topic":"water", "split":split}
            for i,split in enumerate(["train", "validation", "test"])]
    if change:
        change(rows)
    path = tmp_path / "topic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_topic_preserves_supplied_assignments_and_normalizes(tmp_path):
    dataset = build_topic_dataset(topic_csv(tmp_path))
    assert dataset["train"]["feedback_id"] == ["F0"]
    assert dataset["validation"]["feedback_id"] == ["F1"]
    assert dataset["test"]["feedback_id"] == ["F2"]
    assert dataset["train"][0]["text"] == "الخدمة جيدة"


def test_topic_rejects_citizen_overlap(tmp_path):
    path = topic_csv(tmp_path, lambda rows: rows[2].update(citizen_group_id="C0"))
    with pytest.raises(ValueError, match="Citizen overlap"):
        build_topic_dataset(path)


def test_topic_rejects_unknown_evaluation_label(tmp_path):
    path = topic_csv(tmp_path, lambda rows: rows[2].update(topic="unseen"))
    with pytest.raises(ValueError, match="absent from training"):
        build_topic_dataset(path)


def test_fixed_class_macro_f1_reports_missing_class_coverage():
    result = topic_metrics([0, 1, 0, 1], [0, 1, 0, 1], num_labels=4)
    assert result["accuracy"] == 1
    assert result["macro_f1"] == .5
    assert result["num_labels_with_support"] == 2
    assert result["num_task_labels"] == 4


def test_json_evidence_accepts_seqeval_numpy_support_counts(tmp_path):
    from seqeval.metrics import classification_report
    from seqeval.scheme import IOB2
    report = classification_report([["B-SERVICE", "I-SERVICE", "O"]],
                                   [["B-SERVICE", "I-SERVICE", "O"]],
                                   mode="strict", scheme=IOB2, output_dict=True, zero_division=0)
    output = tmp_path / "evidence.json"
    write_json(output, {"entities": report, "array": np.array([1, 2])})
    actual = json.loads(output.read_text())
    assert actual["entities"]["SERVICE"]["support"] == 1
    assert actual["entities"]["SERVICE"]["f1-score"] == 1
    assert actual["array"] == [1, 2]


def test_compound_conll_cell_keeps_entity_boundary(tmp_path):
    path = tmp_path / "ner.conll"
    path.write_text("خدمات المياه\tB-SERVICE\nفي\tO\nالرياض\tB-LOCATION\n", encoding="utf-8")
    result = read_conll(path)
    assert result[0]["tokens"] == ["خدمات", "المياه", "في", "الرياض"]
    assert result[0]["tags"] == ["B-SERVICE", "I-SERVICE", "O", "B-LOCATION"]


def test_ner_template_split_blocks_reference_only_duplicates():
    records = [{"id": f"{city}-{ref}", "tokens":[f"city{city}", f"REF{ref}"],
                "tags":["B-LOCATION","B-REFERENCE"]}
               for city in range(20) for ref in range(3)]
    result = split_ner(records)
    assert sum(map(len, result.values())) == 60
    membership = {}
    for split, rows in result.items():
        for row in rows:
            city = row["tokens"][0]
            assert city not in membership or membership[city] == split
            membership[city] = split
    assert result == split_ner(records)


def test_alignment_rejects_bad_word_indices():
    with pytest.raises(ValueError):
        align_labels([None, 2], [1])
    with pytest.raises(ValueError):
        align_labels([1, 0], [1, 2])


def test_qa_excludes_question_tokens_before_topk_selection():
    result = best_span([100, 4, 1], [100, 1, 5], [None, (0, 4), (5, 9)],
                       null_score=0, null_threshold=0, top_k=1)
    assert result["answer"] == (0, 9)


def test_qa_enforces_answer_length():
    result = best_span([9, 1, 0], [0, 1, 10], [(0, 2), (3, 5), (6, 8)],
                       null_score=0, null_threshold=0, max_answer_len=1)
    assert result["answer"] == (6, 8)


def test_qa_cannot_bridge_non_context_tokens():
    result = best_span([9, 0, 0], [0, 0, 10], [(0, 2), None, (4, 6)],
                       null_score=0, null_threshold=0)
    assert result["answer"] == (4, 6)


def test_qa_no_valid_offsets_returns_null():
    result = best_span([3, 2], [4, 5], [None, (0, 0)], null_score=0, null_threshold=100)
    assert result["answer"] is None


def test_qa_null_threshold_boundary():
    args = ([2], [3], [(0, 4)])
    assert best_span(*args, null_score=6, null_threshold=1)["answer"] == (0, 4)
    assert best_span(*args, null_score=6.01, null_threshold=1)["answer"] is None


def test_qa_ignores_nonfinite_candidates():
    result = best_span([np.nan, 2], [np.nan, 3], [(0, 2), (3, 5)], null_score=0, null_threshold=0)
    assert result["answer"] == (3, 5)


def test_qa_protocol_has_disjoint_contexts_and_real_nulls():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("qa_smoke", root / "scripts/qa_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    supplied = module.read_squad(root / "data/eval/qa_smoke_set.json")
    development, supplemental = module.protocol(module.read_squad(root / "data/models/bayan_qa.json"), supplied)
    assert len(supplemental) == 12
    assert sum(r["is_impossible"] for r in supplemental) == 3
    assert {r["context"] for r in development}.isdisjoint(r["context"] for r in supplied + supplemental)
