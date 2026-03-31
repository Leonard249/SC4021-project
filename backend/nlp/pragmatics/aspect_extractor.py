"""
aspect_extractor.py
SC4021 Information Retrieval 2026 — Aspect Extraction Module

Pipeline position:
    MicrotextNormalizer → SBD → POSTagger → NERTagger → SubjectivityDetector
    → AspectExtractor ← here → SarcasmDetector → LengthAwareRouting → Ensemble

Reads (per record):
    Subjectivity            — record-level label; only "subjective" records are processed
    Subjectivity_Sentences  — list of per-sentence dicts produced by SubjectivityDetector
    NER_Tags                — list of [entity_text, label, start_char, end_char]
    Normalized_Text         — full normalized text (used to reconstruct sentence boundaries)
    Sentences               — pre-split sentence strings (from SBD)

Writes:
    Targeted_Aspects — list of aspect dicts (see format below), added only when
                       the record (or comment) is subjective. Skipped records get
                       an empty list.

Output format (one entry per entity × sentence match):
    {
        "Aspect_Name":       "VS Code",
        "Entity_Type":       "EDITOR",
        "Target_Sentence":   "I've always loved Cline but...",
        "Sentence_Word_Count": 14
    }

Aspect extraction logic
-----------------------
NER spans use Python's native half-open interval convention [start, end), matching
spaCy's ent.start_char / ent.end_char exactly:

    entity  = ["VS Code", "EDITOR", 162, 169]   → text[162:169] == "VS Code"
    length  = end - start  (169 - 162 = 7)

Sentence boundaries are reconstructed by scanning Normalized_Text for each
sentence string and recording its [sent_start, sent_end) interval.  An entity
is linked to a sentence when:

    entity_start >= sent_start  AND  entity_end <= sent_end

This is a strict containment check — the entity span must lie entirely within
the sentence span.  Because both use the same half-open convention the check is
a single two-way inequality, with no off-by-one correction needed.

Only entity types in ASPECT_ENTITY_TYPES are extracted (irrelevant statistical
labels such as PERSON, GPE, DATE are skipped).

Requires: no additional dependencies beyond the standard library.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Only entities with these labels are considered meaningful aspects for
# AI coding productivity sentiment analysis.
ASPECT_ENTITY_TYPES: frozenset[str] = frozenset({
    "AI_TOOL",       # Copilot, ChatGPT, Claude, Cursor …
    "EDITOR",        # VS Code, Neovim, PyCharm …
    "PL",            # Python, TypeScript, Rust …
    "ORG",           # OpenAI, Microsoft, Anthropic …
    "TECH_CONCEPT",  # RAG, fine-tuning, autocomplete …
    "PRODUCT",       # GitHub, npm, Docker …
})

class AspectExtractor:
    """
    Aspect extractor for the SC4021 pipeline.

    Reads NER_Tags and Subjectivity_Sentences from each record, links named
    entities to the subjective sentences they appear in, and writes the result
    to Targeted_Aspects.

    Only records (and comments) whose Subjectivity == "subjective" are
    processed; objective records receive an empty Targeted_Aspects list.

    Usage
    -----
    extractor = AspectExtractor()
    record = extractor.extract_record(record)

    For a full corpus:
    records = extractor.extract_corpus(records)
    """

    def __init__(
        self,
        aspect_entity_types: Optional[frozenset[str]] = None,
    ):
        """
        Parameters
        ----------
        aspect_entity_types : set of NER labels to treat as aspects.
            Defaults to ASPECT_ENTITY_TYPES defined at module level.
        """
        self.aspect_entity_types: frozenset[str] = (
            aspect_entity_types if aspect_entity_types is not None
            else ASPECT_ENTITY_TYPES
        )

    def extract_record(self, record: dict) -> dict:
        """
        Extract aspects for a single record and all its comments in-place.

        Adds 'Targeted_Aspects' to the record and to each comment.
        Returns the modified record.
        """
        record["Targeted_Aspects"] = self._extract(record)

        for comment in record.get("Comments") or []:
            comment["Targeted_Aspects"] = self._extract(comment)

        return record

    def extract_corpus(self, records: list[dict]) -> list[dict]:
        """Extract aspects for an entire list of records."""
        total = len(records)
        for i, record in enumerate(records, 1):
            try:
                self.extract_record(record)
            except Exception as e:
                logger.error(
                    f"AspectExtractor failed on record "
                    f"{record.get('ID', i)}: {e}"
                )
            if i % 500 == 0:
                logger.info(f"Aspect extraction: {i}/{total} records processed.")
        logger.info(
            f"Aspect extraction complete. {total} records processed."
        )
        return records

    def _extract(self, container: dict) -> list[dict]:
        """
        Extract targeted aspects from a single record or comment dict.

        Returns an empty list immediately if:
          - The container is not labelled "subjective", OR
          - NER_Tags is missing / empty, OR
          - Sentences is missing / empty.
        """
        # Gate: skip objective / neutral containers entirely.
        if container.get("Subjectivity", "objective") != "subjective":
            logger.debug(
                f"Skipping record {container.get('ID', '?')} "
                "(not subjective)."
            )
            return []

        ner_tags: list[list] = container.get("NER_Tags") or []
        sentences: list[str] = container.get("Sentences") or []
        normalized_text: str = container.get("Normalized_Text", "")

        if not ner_tags or not sentences:
            return []

        # Build [sent_start, sent_end) intervals for every sentence.
        sent_intervals = self._build_sentence_intervals(normalized_text, sentences)

        aspects: list[dict] = []
        seen: set[tuple] = set()  # deduplicate (entity_text, sent_start) pairs

        for entity in ner_tags:
            # NER_Tags entry: [entity_text, label, start_char, end_char]
            if len(entity) < 4:
                continue

            entity_text, entity_label, ent_start, ent_end = (
                entity[0], entity[1], entity[2], entity[3]
            )

            # Skip entity types that are not relevant aspects.
            if entity_label not in self.aspect_entity_types:
                continue

            # Find the sentence whose [sent_start, sent_end) contains this entity.
            matched_sentence = self._match_sentence(
                ent_start, ent_end, sent_intervals, sentences
            )
            if matched_sentence is None:
                logger.debug(
                    f"Entity '{entity_text}' [{ent_start}:{ent_end}] "
                    "did not match any sentence interval — skipped."
                )
                continue

            sent_text, sent_start = matched_sentence

            # Deduplicate: same entity mention in the same sentence once only.
            key = (entity_text.lower(), sent_start)
            if key in seen:
                continue
            seen.add(key)

            aspects.append({
                "Aspect_Name":        entity_text,
                "Entity_Type":        entity_label,
                "Target_Sentence":    sent_text,
                "Sentence_Word_Count": len(sent_text.split()),
            })

        return aspects

    # Sentence interval reconstruction

    def _build_sentence_intervals(
        self,
        normalized_text: str,
        sentences: list[str],
    ) -> list[tuple[int, int]]:
        """
        Reconstruct [start, end) character intervals for each sentence by
        scanning Normalized_Text left-to-right.

        Scans from `search_from` after each match so that repeated sentence
        strings (e.g. "Yes.") are mapped to the correct occurrence.

        Returns a list of (start, end) tuples aligned with `sentences`.
        """
        intervals: list[tuple[int, int]] = []
        search_from = 0

        for sentence in sentences:
            if not sentence:
                intervals.append((search_from, search_from))
                continue

            idx = normalized_text.find(sentence, search_from)
            if idx == -1:
                # Sentence not found at expected position — use a best-effort
                # global search from the beginning (handles minor whitespace
                # discrepancies introduced by the normalizer).
                idx = normalized_text.find(sentence)

            if idx == -1:
                # Still not found: append a sentinel so list stays aligned.
                logger.warning(
                    f"Could not locate sentence in Normalized_Text: "
                    f"'{sentence[:60]}...'"
                )
                intervals.append((search_from, search_from))
            else:
                end = idx + len(sentence)
                intervals.append((idx, end))
                search_from = end  # advance cursor past this sentence

        return intervals

    def _match_sentence(
        self,
        ent_start: int,
        ent_end: int,
        sent_intervals: list[tuple[int, int]],
        sentences: list[str],
    ) -> Optional[tuple[str, int]]:
        """
        Return (sentence_text, sentence_start) for the first sentence whose
        [sent_start, sent_end) interval strictly contains [ent_start, ent_end).

        The containment condition is:
            ent_start >= sent_start  AND  ent_end <= sent_end

        Both spans use the same half-open [inclusive, exclusive) convention,
        so no off-by-one adjustment is required.
        """
        for (sent_start, sent_end), sent_text in zip(sent_intervals, sentences):
            if ent_start >= sent_start and ent_end <= sent_end:
                return sent_text, sent_start
        return None

def load_json(path: str | Path) -> list[dict] | dict:
    """Load a JSON file; returns a list or a single dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list[dict] | dict, path: str | Path) -> None:
    """Save data to a JSON file with readable indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved output to {path}")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_record = {
        "ID": "r_001",
        "Source": "Reddit",
        "Subjectivity": "subjective",
        "Subjectivity_Score": 0.72,
        "Normalized_Text": (
            "I added Cline to Zed Like the title says, I built an ACP bridge for the Cline CLI so that I can use Cline with Zed. "
            "I've always loved Cline but obviously being forced into VS Code was a bummer. "
            "With Claude Opus 4.5, I gave it a shot translating the Claude Code ACP plugin from Zed to work with the Cline CLI. "
            "It took some time and no doubt has bugs, but it's pretty cool what standards do. "
            "The native GUI elements work for following the agent, selecting the mode, and selecting the model."
        ),
        "Sentences": [
            "I added Cline to Zed Like the title says, I built an ACP bridge for the Cline CLI so that I can use Cline with Zed.",
            "I've always loved Cline but obviously being forced into VS Code was a bummer.",
            "With Claude Opus 4.5, I gave it a shot translating the Claude Code ACP plugin from Zed to work with the Cline CLI. It took some time and no doubt has bugs, but it's pretty cool what standards do.",
            "The native GUI elements work for following the agent, selecting the mode, and selecting the model.",
        ],
        "NER_Tags": [
            ["VS Code",        
             "EDITOR",   
             172, 
             179],
            ["Claude",         
             "AI_TOOL",  
             199, 
             205],
            ["Claude",         
             "AI_TOOL",  
             249, 
             255],
            ["ACP plugin",    
             "AI_TOOL",
             261, 
             271],
            ["the Cline CLI",  
             "ORG",      
             68, 
             81],
        ],
        "Subjectivity_Sentences": [
            {"text": "I added Cline to Zed Like the title says, I built an ACP bridge for the Cline CLI so that I can use Cline with Zed.", "label": "objective", "score": 0.39, "method": "transformer"},
            {"text": "I've always loved Cline but obviously being forced into VS Code was a bummer.", "label": "subjective", "score": 0.72, "method": "transformer"},
            {"text": "With Claude Opus 4.5, I gave it a shot translating the Claude Code ACP plugin from Zed to work with the Cline CLI. It took some time and no doubt has bugs, but it's pretty cool what standards do.", "label": "subjective", "score": 0.54, "method": "transformer"},
            {"text": "The native GUI elements work for following the agent, selecting the mode, and selecting the model.", "label": "objective", "score": 0.0, "method": "lexicon"},
        ],
        "Comments": [
            {
                "comment_id": "c_001",
                "Subjectivity": "objective",
                "Normalized_Text": "Kilocode has officially released its command line interface.",
                "Sentences": ["Kilocode has officially released its command line interface."],
                "NER_Tags": [["Kilocode", "PERSON", 0, 8]],
                "Subjectivity_Sentences": [
                    {"text": "Kilocode has officially released its command line interface.", "label": "objective", "score": 0.0, "method": "lexicon"}
                ],
            }
        ],
    }
    
    dataset_path = '../../../data/my_test/sample.json'
    data = load_json(dataset_path)
    records: list[dict] = data if isinstance(data, list) else [data]

    extractor = AspectExtractor()
    extractor.extract_corpus(records)

    result_to_print = records if len(records) > 1 else records[0]
    print(json.dumps(result_to_print, indent=4))