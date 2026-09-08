"""Topic data with immutable supplied splits and explicit leakage checks."""
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from bayan.preprocessing.core import preprocess

ROOT = Path(__file__).resolve().parents[3]
TOPIC_DATA = ROOT / "data/raw/bayan_feedback.csv"
SPLITS = ("train", "validation", "test")


def build_topic_dataset(path=TOPIC_DATA):
    """Preserve the supplied frozen split; fail rather than silently resplit.

    Label vocabulary is learned from train only. Each example retains its ID,
    citizen group and original metadata for later reproducible sliced reports.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = ["feedback_id", "citizen_group_id", "text", "topic", "split"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Required columns: {required}")
    if frame[required].isna().any().any() or frame[required].astype(str).apply(lambda c: c.str.strip().eq("")).any().any():
        raise ValueError("Required fields must not be missing or blank")
    if frame.feedback_id.duplicated().any():
        raise ValueError("feedback_id must be unique")
    if set(frame.split) != set(SPLITS):
        raise ValueError("Expected nonempty train, validation and test splits")
    if frame.groupby("citizen_group_id").split.nunique().gt(1).any():
        raise ValueError("Citizen overlap across splits; repair source data, not the frozen test")
    labels = sorted(frame.loc[frame.split.eq("train"), "topic"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    if not set(frame.topic).issubset(label2id):
        raise ValueError("Evaluation contains a topic absent from training")
    frame["label"] = frame.topic.map(label2id).astype(int)
    frame["text"] = frame.text.map(preprocess)
    if frame.text.eq("").any():
        raise ValueError("Preprocessing produced an empty text")
    return DatasetDict({
        split: Dataset.from_pandas(frame.loc[frame.split.eq(split)].reset_index(drop=True), preserve_index=False)
        for split in SPLITS
    })
