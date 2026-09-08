"""NER word-to-subword alignment and tab-delimited CoNLL parsing."""
from pathlib import Path
import hashlib
import json

from sklearn.model_selection import GroupShuffleSplit

TAGS = ["O"] + [f"{prefix}-{entity}" for entity in
                 ("SERVICE", "LOCATION", "DATE", "REFERENCE", "ORGANISATION")
                 for prefix in ("B", "I")]


def align_labels(word_ids, word_labels):
    """Supervise only the first subword of each word; ignore special/pad pieces."""
    aligned = []
    seen = set()
    previous = -1
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
            continue
        if not isinstance(word_id, int) or not 0 <= word_id < len(word_labels):
            raise ValueError("word_id is outside the word-label sequence")
        if word_id < previous:
            raise ValueError("Expected a single sequence with monotonic word_ids")
        aligned.append(-100 if word_id in seen else word_labels[word_id])
        seen.add(word_id)
        previous = word_id
    return aligned


def read_conll(path):
    """Expand whitespace inside a source cell, preserving its BIO entity span.

    The supplied file contains entries such as 'خدمات المياه<TAB>B-SERVICE'.
    Splitting only on the last TAB preserves the label before expanding this
    into two words tagged B-SERVICE, I-SERVICE for faithful entity evaluation.
    """
    records, words, tags = [], [], []

    def flush():
        if words:
            records.append({"id": f"NER-{len(records):06d}", "tokens": words.copy(), "tags": tags.copy()})
            words.clear()
            tags.clear()

    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            flush()
            continue
        if "\t" not in line:
            raise ValueError(f"Expected TAB-delimited token/tag at line {number}")
        token, tag = line.rsplit("\t", 1)
        tag = tag.strip()
        pieces = token.split()
        if not pieces or tag not in TAGS:
            raise ValueError(f"Invalid token or BIO tag at line {number}")
        words.extend(pieces)
        tags.extend([tag] + [("I-" + tag[2:]) if tag.startswith("B-") else tag] * (len(pieces) - 1))
    flush()
    if not records:
        raise ValueError("Empty CoNLL dataset")
    return records


def split_ner(records, seed=42):
    """Keep repeated templates together, ignoring the unique reference value.

    No citizen IDs or split column exist in the NER file. A template is the
    word/tag sequence with REFERENCE values replaced by their entity type;
    this prevents near-identical sentences with different IDs crossing splits.
    """
    groups = []
    for row in records:
        key = [("<REFERENCE>" if tag.endswith("-REFERENCE") else word, tag)
               for word, tag in zip(row["tokens"], row["tags"])]
        groups.append(hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest())
    indices = list(range(len(records)))
    train_valid, test = next(GroupShuffleSplit(n_splits=1, test_size=.1, random_state=seed).split(indices, groups=groups))
    train_sub, valid_sub = next(GroupShuffleSplit(n_splits=1, test_size=2/9, random_state=seed).split(
        train_valid, groups=[groups[i] for i in train_valid]))
    split_indices = {"train": train_valid[train_sub], "validation": train_valid[valid_sub], "test": test}
    result = {name: [{**records[i], "template_group": groups[i]} for i in idx] for name, idx in split_indices.items()}
    sets = {name: {r["template_group"] for r in rows} for name, rows in result.items()}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
    return result
