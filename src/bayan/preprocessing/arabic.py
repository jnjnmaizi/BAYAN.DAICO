"""Versioned Arabic profiles and CAMeL D3 clitics with source-word alignment."""
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
import unicodedata

from .core import mask_pii

ARABIC_PREPROC_VERSION = "1.0.0"
_FOLD = str.maketrans({**dict.fromkeys("أإآٱ", "ا"), "ؤ": "و", "ئ": "ي", "ى": "ي", "ة": "ه"})
_ARABIC = re.compile(r"[\u0621-\u063a\u0641-\u064a\u066e-\u06d3]")


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False

    def __post_init__(self):
        if self.name not in {"bayan_ar_v1", "camelbert_v1"}:
            raise ValueError(f"Unknown Arabic profile: {self.name}")


BAYAN_AR_V1 = ArabicProfile("bayan_ar_v1", dediacritize=True)
CAMELBERT_V1 = ArabicProfile("camelbert_v1", dediacritize=True)


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    """Normalize model text; preserve case, emoji and repeated letters.

    bayan_ar_v1 follows the supplied golden pairs. camelbert_v1 is our named
    conservative profile: no letter folding, following CAMeLBERT's model card.
    """
    if not isinstance(text, str):
        raise TypeError("Arabic normalization expects a string")
    from camel_tools.utils.dediac import dediac_ar
    text = unicodedata.normalize("NFC", text).replace("ـ", "")
    if profile.dediacritize:
        text = dediac_ar(text)
    if profile.name == "bayan_ar_v1":
        text = text.translate(_FOLD)
    return " ".join(text.split())


def prepare_arabic_text(text: str, profile: ArabicProfile) -> dict[str, str]:
    """Retain original spelling for display while masking PII in both copies."""
    display = mask_pii(text)
    return {"display_text": display, "model_text": normalize_arabic(display, profile),
            "profile": profile.name, "version": ARABIC_PREPROC_VERSION}


@lru_cache(maxsize=1)
def _d3_tokenizer():
    # Keep large resources local to the project unless the caller selects a path.
    default = Path(__file__).resolve().parents[3] / "artifacts/camel_tools"
    path = Path(os.environ.setdefault("CAMELTOOLS_DATA", str(default)))
    if not (path / "catalogue.json").is_file():
        raise RuntimeError("CAMeL data missing. Run: python scripts/setup_lab4.py")
    try:
        from camel_tools.disambig.mle import MLEDisambiguator
        from camel_tools.tokenizers.morphological import MorphologicalTokenizer
        return MorphologicalTokenizer(MLEDisambiguator.pretrained("calima-msa-r13"),
                                      scheme="d3tok", split=False, diac=False)
    except (OSError, KeyError) as error:
        raise RuntimeError("CAMeL MSA resources missing. Run: python scripts/setup_lab4.py") from error


@lru_cache(maxsize=8192)
def _segment_word(word: str) -> tuple[str, ...]:
    if not _ARABIC.search(word):
        return (word,)
    output = _d3_tokenizer().tokenize([word])
    if len(output) != 1:
        raise ValueError("Expected one D3 analysis per source word")
    return (word,) if output[0] == word else tuple(output[0].split("_"))


def segment_words(words: list[str]) -> list[list[str]]:
    """One group per input word: preserve English, dates and IDs verbatim."""
    if any(not isinstance(word, str) or not word for word in words):
        raise ValueError("Expected nonempty source words")
    return [list(_segment_word(word)) for word in words]


def segment(text: str) -> list[str]:
    """Human-readable D3 tokens, e.g. وبالرياض -> و+ ب+ ال+ رياض."""
    from camel_tools.tokenizers.word import simple_word_tokenize
    return [piece for group in segment_words(simple_word_tokenize(text)) for piece in group]


def ner_segmented_view(words: list[str]) -> tuple[list[str], list[int]]:
    """Return D3 pieces and one lexical-stem position per original NER word.

    The mapping uses morphology only, never gold labels or model predictions.
    Project predictions back through these positions for comparable BIO spans.
    """
    pieces, anchors = [], []
    for group in segment_words(words):
        lexical = [i for i, piece in enumerate(group)
                   if not piece.startswith("+") and not piece.endswith("+")]
        if len(lexical) != 1:
            raise ValueError(f"Expected one lexical stem in D3 group: {group}")
        anchors.append(len(pieces) + lexical[0])
        pieces.extend(group)
    return pieces, anchors
