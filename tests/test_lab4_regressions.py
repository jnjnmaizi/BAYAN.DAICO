"""Lab 4 profiles, morphology alignment, and dialect audit contracts."""
import importlib.util
from pathlib import Path

import pytest

from bayan.preprocessing import arabic
from bayan.preprocessing.arabic import (ArabicProfile, BAYAN_AR_V1, CAMELBERT_V1,
                                       normalize_arabic, prepare_arabic_text)


def test_model_profiles_keep_distinct_letter_policies():
    text = "إضاءة خِدْمَة على مسؤول"
    assert normalize_arabic(text, BAYAN_AR_V1) == "اضاءه خدمه علي مسوول"
    assert normalize_arabic(text, CAMELBERT_V1) == "إضاءة خدمة على مسؤول"


def test_normalization_keeps_signal_and_does_not_recollapse_folded_alefs():
    assert normalize_arabic("أإآا  😍 BYN-ABC", BAYAN_AR_V1) == "اااا 😍 BYN-ABC"


@pytest.mark.parametrize("profile", [BAYAN_AR_V1, CAMELBERT_V1])
def test_arabic_normalization_is_idempotent(profile):
    normalized = normalize_arabic("  إضااااءةُ مـدرسة  ", profile)
    assert normalize_arabic(normalized, profile) == normalized


def test_diacritic_option_can_preserve_vowels():
    assert normalize_arabic("خِدْمَة", ArabicProfile("camelbert_v1")) == "خِدْمَة"


def test_display_spelling_is_preserved_with_pii_masked_in_both_views():
    raw = "إضاءةُ  مـدرسة 0551234567"
    view = prepare_arabic_text(raw, BAYAN_AR_V1)
    assert view["display_text"] == "إضاءةُ  مـدرسة <PHONE>"
    assert view["model_text"] == "اضاءه مدرسه <PHONE>"
    assert "0551234567" not in str(view)
    assert raw == "إضاءةُ  مـدرسة 0551234567"


def test_unknown_profile_is_not_silently_treated_as_another_model():
    with pytest.raises(ValueError, match="Unknown Arabic profile"):
        ArabicProfile("misspelled-profile")


def test_non_arabic_ner_words_do_not_require_morphology_or_split_identifiers():
    words = ["Water_Service", "2025-05-12", "BYN-000001"]
    assert arabic.segment_words(words) == [[word] for word in words]


def test_ner_projection_keeps_source_word_boundaries(monkeypatch):
    monkeypatch.setattr(arabic, "segment_words", lambda words:[["و+", "ب+", "ال+", "رياض"], ["مرجع", "+ه"]])
    pieces, anchors = arabic.ner_segmented_view(["وبالرياض", "مرجعه"])
    assert [pieces[i] for i in anchors] == ["رياض", "مرجع"]
    assert len(anchors) == 2


def test_ner_projection_rejects_ambiguous_stem_mapping(monkeypatch):
    monkeypatch.setattr(arabic, "segment_words", lambda words:[["stem", "another"]])
    with pytest.raises(ValueError, match="one lexical stem"):
        arabic.ner_segmented_view(["word"])


def test_dialect_audit_uses_only_arabic_metadata(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("dialect_audit", root / "scripts/dialect_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / "feedback.csv"
    data.write_text("lang,dialect_region,split\nar,Gulf,train\nar,MSA,validation\nen,Gulf,test\n")
    result = module.audit(data)
    assert result["counts"] == {"Gulf":1,"MSA":1}
    assert result["percent_of_arabic"] == {"Gulf":50,"MSA":50}
    assert result["by_split"]["test"] == {}


def test_dialect_audit_requires_nonempty_arabic_labels(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("dialect_audit", root / "scripts/dialect_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / "feedback.csv"
    data.write_text("lang,dialect_region,split\nar,,train\n")
    with pytest.raises(ValueError, match="nonempty dialect"):
        module.audit(data)


def test_bakeoff_marks_absent_gulf_slice_unavailable():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("arabic_bakeoff", root / "scripts/arabic_bakeoff.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.slice_metrics([{"dialect_region":"MSA","label":0}], [0], 2)
    assert result["Gulf"] is None
    assert result["MSA"]["macro_f1"] == .5


def test_bert_strided_weights_can_be_saved_and_reloaded(tmp_path):
    import torch
    from transformers import BertConfig, BertForSequenceClassification, TrainingArguments
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("arabic_bakeoff", root / "scripts/arabic_bakeoff.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = BertForSequenceClassification(BertConfig(vocab_size=12, hidden_size=8,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=16))
    weight = model.bert.encoder.layer[0].attention.self.query.weight
    weight.data = weight.data.t()
    assert not weight.is_contiguous()
    trainer = module.ContiguousTrainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to=[]))
    trainer.save_model(str(tmp_path))
    restored = BertForSequenceClassification.from_pretrained(tmp_path, local_files_only=True)
    assert torch.equal(restored.bert.encoder.layer[0].attention.self.query.weight, weight)
