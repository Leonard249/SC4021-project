"""
ner_tagger.py
SC4021 Information Retrieval 2026 — NER Tagging Module

Pipeline position:
    MicrotextNormalizer → SBD → POSTagger → NERTagger ← here → SubjectivityDetector

Reads:
    Normalized_Text — full normalized text (from MicrotextNormalizer)

Writes:
    NER_Tags — list of named entity spans detected in Normalized_Text.

NER_Tags storage format:
    [
        ["GitHub Copilot", "AI_TOOL", 0,  14],
        ["Microsoft",      "ORG",     34, 43],
        ["Python",         "PL",      58, 64],
    ]

    Each entry: [entity_text, label, start_char, end_char]
    Character offsets are relative to Normalized_Text.

    NER intentionally operates on the full Normalized_Text (not sentence-by-
    sentence like POS), because multi-sentence context improves entity
    disambiguation for the statistical NER model.

Standard spaCy NER labels used:
    PERSON, ORG, GPE, DATE, CARDINAL, TIME, ...

Custom labels added via EntityRuler (domain-specific, defined in spacy_utils):
    AI_TOOL  -- AI coding assistants (Copilot, ChatGPT, Claude, Cursor ...)
    EDITOR   -- Code editors and IDEs (VS Code, Neovim, PyCharm ...)
    PL       -- Programming languages (Python, TypeScript, Rust ...)

    <CODE> and [emoticon] tokens are filtered OUT of NER_Tags --
    they are already captured in POS_Tags with CODE/EMOTICON labels.

Requires:
    pip install spacy
    python -m spacy download en_core_web_sm
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2] 
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import logging
from typing import Optional
import spacy

from utils.spacy_utils import _extract_special_tokens, DOMAIN_ENTITY_PATTERNS, DYNAMIC_POS_PATTERNS

logger = logging.getLogger(__name__)


class NERTagger:
    """
    Named entity recogniser for the SC4021 pipeline.

    Combines spaCy's statistical NER model with a domain-specific EntityRuler
    covering AI tools, editors, and programming languages not in spaCy's vocab.

    The EntityRuler runs AFTER the statistical NER model (overwrite_ents=True),
    so curated domain patterns always win over spaCy's predictions.

    Usage
    -----
    ner = NERTagger()
    record = ner.tag_record(record)   # requires Normalized_Text

    For a full corpus (faster):
    records = ner.tag_corpus(records)
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        extra_patterns: Optional[list[dict]] = None,
    ):
        """
        Parameters
        ----------
        model : spaCy model name (same options as POSTagger).
        extra_patterns : additional EntityRuler patterns merged with
            DOMAIN_ENTITY_PATTERNS from spacy_utils.
            Format: [{"label": "AI_TOOL", "pattern": [{"LOWER": "mytool"}]}, ...]
        """
        try:
            self._nlp = spacy.load(model)
            logger.info(f"NERTagger: loaded spaCy model '{model}'")
        except OSError:
            raise OSError(
                f"spaCy model '{model}' not found. "
                f"Install with: python -m spacy download {model}"
            )

        # Keep tok2vec and ner. Disable POS/lemma pipes since POS tagging is handled separately by POSTagger.
        self._nlp.disable_pipes(
            [p for p in ["lemmatizer", "parser"]
             if p in self._nlp.pipe_names]

            # [p for p in ["tagger", "attribute_ruler", "lemmatizer", "parser"]
            #  if p in self._nlp.pipe_names]
        )

        # EntityRuler runs AFTER statistical NER so domain patterns overwrite.
        ruler = self._nlp.add_pipe(
            "entity_ruler",
            after="ner",
            config={"overwrite_ents": True},
        )
        patterns = DOMAIN_ENTITY_PATTERNS + DYNAMIC_POS_PATTERNS + (extra_patterns or [])
        ruler.add_patterns(patterns)

        logger.info(f"NERTagger: EntityRuler loaded {len(patterns)} domain patterns")
        logger.info(f"NERTagger: active pipes: {self._nlp.pipe_names}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag_record(self, record: dict) -> dict:
        """
        NER-tag a single record and all its comments in-place.

        Requires 'Normalized_Text' (from MicrotextNormalizer).
        Adds 'NER_Tags' -- a list of [text, label, start_char, end_char].

        Returns the modified record.
        """
        text = record.get("Normalized_Text", "")
        if not text:
            logger.warning(
                f"Record {record.get('ID', '?')} has no Normalized_Text -- skipping."
            )
            record["NER_Tags"] = []
            return record

        record["NER_Tags"] = self._tag_text(text)

        for comment in record.get("Comments") or []:
            c_text = comment.get("Normalized_Text", "")
            comment["NER_Tags"] = self._tag_text(c_text) if c_text else []

        return record

    def tag_corpus(self, records: list[dict]) -> list[dict]:
        """
        NER-tag an entire corpus using spaCy's batch pipe for speed.
        """
        targets: list[tuple[dict, str]] = []
        texts: list[str] = []

        for record in records:
            targets.append((record, "NER_Tags"))
            texts.append(record.get("Normalized_Text", ""))
            for comment in record.get("Comments") or []:
                targets.append((comment, "NER_Tags"))
                texts.append(comment.get("Normalized_Text", ""))

        if not texts:
            return records

        cleaned_texts, special_maps = zip(
            *[_extract_special_tokens(t) for t in texts]
        )

        logger.info(f"NERTagger: batch tagging {len(texts)} texts...")
        docs = list(self._nlp.pipe(cleaned_texts, batch_size=256))

        for (container, field), doc, special_map in zip(targets, docs, special_maps):
            container[field] = self._build_ner_list(doc, special_map)

        logger.info("NERTagger: batch tagging complete.")
        return records

    # ------------------------------------------------------------------
    # Core NER logic
    # ------------------------------------------------------------------

    def _tag_text(self, text: str) -> list[list]:
        """NER-tag a single text string."""
        cleaned, special_map = _extract_special_tokens(text)
        doc = self._nlp(cleaned)
        return self._build_ner_list(doc, special_map)

    def _build_ner_list(
        self,
        doc,
        special_map: dict[str, tuple[str, str]],
    ) -> list[list]:
        """
        Build the NER_Tags list from a spaCy Doc.

        Filters out CODE and EMOTICON pipeline tokens -- those are already
        captured in POS_Tags and do not belong in NER_Tags.
        """
        result: list[list] = []

        VALID_LABELS = {
            "AI_TOOL", "EDITOR", "PL", "ORG", "TECH_CONCEPT", 
            "PRODUCT", "PERSON", "EVENT", "GPE"
        }

        for ent in doc.ents:
            surface = ent.text

            if surface in special_map:
                _, custom_label = special_map[surface]
                # Skip CODE and EMOTICON -- already in POS_Tags
                if custom_label in ("CODE", "EMOTICON"):
                    continue
                # Any other custom-labelled special token (unlikely but safe)
                result.append([surface, custom_label, ent.start_char, ent.end_char])
            else:

                # Drop the entity if it is a junk statistical label (like TIME or CARDINAL)
                if ent.label_ not in VALID_LABELS:
                    continue

                # Restore any placeholders embedded inside a multi-token span
                restored = surface
                for placeholder, (original, _) in special_map.items():
                    restored = restored.replace(placeholder, original)
                result.append([restored, ent.label_, ent.start_char, ent.end_char])

        return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_record = {
        "ID": "r_001",
        "Source": "Reddit",
        "Normalized_Text": (
            "GitHub Copilot and ChatGPT are both great tools. "
            "OpenAI and Microsoft built them. "
            "I use VS Code with Python and TypeScript every day. "
            "Cursor is also worth trying [smiley face]. "
            "Check the docs with <CODE>."
        ),
        "Sentences": [
            "GitHub Copilot and ChatGPT are both great tools.",
            "OpenAI and Microsoft built them.",
            "I use VS Code with Python and TypeScript every day.",
            "Cursor is also worth trying [smiley face].",
            "Check the docs with <CODE>.",
        ],
        "Comments": [
            {
                "comment_id": "c_001",
                "Normalized_Text": "I switched from Tabnine to Cursor last month.",
                "Sentences": ["I switched from Tabnine to Cursor last month."],
            }
        ],
    }

    ner = NERTagger(model="en_core_web_sm")
    result = ner.tag_record(sample_record)

    print("\n=== NER Tags (Post) ===")
    print(f"  {'Entity':<35} {'Label':<12} {'Start':>6} {'End':>6}")
    print(f"  {'-'*35} {'-'*12} {'-'*6} {'-'*6}")
    for text, label, start, end in result["NER_Tags"]:
        print(f"  {text:<35} {label:<12} {start:>6} {end:>6}")

    print("\n=== NER Tags (Comment) ===")
    print(f"  {'Entity':<35} {'Label':<12} {'Start':>6} {'End':>6}")
    print(f"  {'-'*35} {'-'*12} {'-'*6} {'-'*6}")
    for text, label, start, end in result["Comments"][0]["NER_Tags"]:
        print(f"  {text:<35} {label:<12} {start:>6} {end:>6}")