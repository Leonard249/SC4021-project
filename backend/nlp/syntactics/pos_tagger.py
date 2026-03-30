"""
pos_tagger.py
SC4021 Information Retrieval 2026 — POS Tagging Module

Pipeline position:
    MicrotextNormalizer → SBD → POSTagger ← here → NERTagger → SubjectivityDetector

Reads:
    Normalized_Text — full normalized text (from MicrotextNormalizer)
    Sentences       — pre-split sentence list (from SBD)

Writes:
    POS_Tags — sentence-aligned list of content-word triples.

POS_Tags storage format (sentence-aligned, content words only):
    [
        [["copilot", "NOUN", "copilot"], ["great", "ADJ", "great"]],  <- sentence 0
        [["makes", "VERB", "make"], ["faster", "ADJ", "fast"]],        <- sentence 1
    ]

    POS_Tags[i] always corresponds to Sentences[i].

    Only CONTENT_POS tags are stored (NOUN, PROPN, VERB, ADJ, ADV, INTJ, NUM,
    CODE, EMOTICON). Function words, punctuation, and whitespace are excluded
    -- no downstream task needs them, and excluding them reduces size by ~40-50%
    for long blog posts.

    Special pipeline tokens:
        <CODE>         -> ["<CODE>",         "CODE",     "<CODE>"]
        [smiley face]  -> ["[smiley face]",  "EMOTICON", "[smiley face]"]

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
import spacy
from utils.spacy_utils import _extract_special_tokens, CONTENT_POS

logger = logging.getLogger(__name__)


class POSTagger:
    """
    Sentence-aligned POS tagger for the SC4021 pipeline.

    Reads the Sentences field produced by SBD and tags each sentence
    individually, storing results as POS_Tags[i] <-> Sentences[i].

    Usage
    -----
    tagger = POSTagger()
    record = tagger.tag_record(record)   # requires Sentences field

    For a full corpus (faster -- uses spaCy batch processing):
    records = tagger.tag_corpus(records)
    """

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Parameters
        ----------
        model : spaCy model name.
            en_core_web_sm  -- fast, recommended for most cases
            en_core_web_md  -- adds word vectors, better OOV handling
            en_core_web_lg  -- most accurate, slowest
        """
        try:
            self._nlp = spacy.load(model)
            logger.info(f"POSTagger: loaded spaCy model '{model}'")
        except OSError:
            raise OSError(
                f"spaCy model '{model}' not found. "
                f"Install with: python -m spacy download {model}"
            )

        # Keep only pipes needed for POS tagging and lemmatization.
        # Disable parser and NER -- NER runs separately in ner_tagger.py.
        self._nlp.disable_pipes(
            [p for p in ["parser", "ner"] if p in self._nlp.pipe_names]
        )
        logger.info(f"POSTagger: active pipes: {self._nlp.pipe_names}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag_record(self, record: dict) -> dict:
        """
        POS-tag a single record and all its comments in-place.

        Requires 'Sentences' (from SBD) to be present.
        Adds 'POS_Tags' -- a sentence-aligned list where POS_Tags[i]
        contains content-word triples for Sentences[i].

        Returns the modified record.
        """
        sentences = record.get("Sentences")
        if not sentences:
            logger.warning(
                f"Record {record.get('ID', '?')} has no Sentences field. "
                "Run SBD before POSTagger."
            )
            record["POS_Tags"] = []
            return record

        record["POS_Tags"] = self._tag_sentences(sentences)

        for comment in record.get("Comments") or []:
            c_sentences = comment.get("Sentences", [])
            comment["POS_Tags"] = self._tag_sentences(c_sentences) if c_sentences else []

        return record

    def tag_corpus(self, records: list[dict]) -> list[dict]:
        """
        POS-tag an entire corpus using spaCy's batch pipe for speed.

        Flattens all sentences from all records and comments into a single
        batch, tags them in one pass, then writes results back sentence-
        aligned into each record's POS_Tags field.
        """
        targets: list[tuple[dict, int]] = []
        all_sentences: list[str] = []

        for record in records:
            sentences = record.get("Sentences", [])
            if not sentences:
                record["POS_Tags"] = []
            else:
                record["POS_Tags"] = [None] * len(sentences)
                for idx, sent in enumerate(sentences):
                    targets.append((record, idx))
                    all_sentences.append(sent)

            for comment in record.get("Comments") or []:
                c_sentences = comment.get("Sentences", [])
                if not c_sentences:
                    comment["POS_Tags"] = []
                else:
                    comment["POS_Tags"] = [None] * len(c_sentences)
                    for idx, sent in enumerate(c_sentences):
                        targets.append((comment, idx))
                        all_sentences.append(sent)

        if not all_sentences:
            return records

        cleaned_sentences, special_maps = zip(
            *[_extract_special_tokens(s) for s in all_sentences]
        )

        logger.info(f"POSTagger: batch tagging {len(all_sentences)} sentences...")
        docs = list(self._nlp.pipe(cleaned_sentences, batch_size=512))

        for (container, sent_idx), doc, special_map in zip(targets, docs, special_maps):
            container["POS_Tags"][sent_idx] = self._build_tag_list(doc, special_map)

        logger.info("POSTagger: batch tagging complete.")
        return records

    # ------------------------------------------------------------------
    # Core tagging logic
    # ------------------------------------------------------------------

    def _tag_sentences(self, sentences: list[str]) -> list[list[list[str]]]:
        """Tag a list of sentences, returning sentence-aligned content-word triples."""
        result = []
        for sentence in sentences:
            cleaned, special_map = _extract_special_tokens(sentence)
            doc = self._nlp(cleaned)
            result.append(self._build_tag_list(doc, special_map))
        return result

    def _build_tag_list(
        self,
        doc,
        special_map: dict[str, tuple[str, str]],
    ) -> list[list[str]]:
        """
        Convert a spaCy Doc into content-word [token, POS, lemma] triples.
        Pipeline tokens are always kept; function words/punctuation are filtered.
        """
        result: list[list[str]] = []

        for token in doc:
            surface = token.text

            if surface in special_map:
                original, custom_tag = special_map[surface]
                result.append([original, custom_tag, original])
            else:
                pos = token.pos_
                if pos not in CONTENT_POS:
                    continue
                result.append([surface, pos, token.lemma_])

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
            "GitHub Copilot was released by Microsoft in 2021. "
            "Honestly it has made me so much more productive [smiling face]. "
            "I think it is better than tabnine for Python development. "
            "Check out <CODE> for the docs."
        ),
        "Sentences": [
            "GitHub Copilot was released by Microsoft in 2021.",
            "Honestly it has made me so much more productive [smiling face].",
            "I think it is better than tabnine for Python development.",
            "Check out <CODE> for the docs.",
        ],
        "Comments": [
            {
                "comment_id": "c_001",
                "Normalized_Text": "I completely agree. It feels incredibly intuitive.",
                "Sentences": [
                    "I completely agree.",
                    "It feels incredibly intuitive.",
                ],
            }
        ],
    }

    tagger = POSTagger(model="en_core_web_sm")
    result = tagger.tag_record(sample_record)

    print("\n=== POS Tags (Post) -- sentence-aligned, content words only ===")
    for i, (sentence, pos_tags) in enumerate(
        zip(result["Sentences"], result["POS_Tags"])
    ):
        print(f"\n  Sentence {i}: {sentence}")
        for token, pos, lemma in pos_tags:
            print(f"    {token:<35} {pos:<12} {lemma}")

    print("\n=== POS Tags (Comment) ===")
    comment = result["Comments"][0]
    for i, (sentence, pos_tags) in enumerate(
        zip(comment["Sentences"], comment["POS_Tags"])
    ):
        print(f"\n  Sentence {i}: {sentence}")
        for token, pos, lemma in pos_tags:
            print(f"    {token:<35} {pos:<12} {lemma}")