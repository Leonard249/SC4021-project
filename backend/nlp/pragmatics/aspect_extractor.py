"""
aspect_extractor.py
SC4021 Information Retrieval 2026 — Aspect Extraction Module
 
Pipeline position:
    MicrotextNormalizer → SBD → POSTagger → NERTagger → SubjectivityDetector
    → AspectExtractor ← here → SarcasmDetector → LengthAwareRouting → Ensemble
 
Reads (per record):
    Subjectivity            — record-level label; only "subjective" records are processed
    NER_Tags                — list of [entity_text, label, start_char, end_char]
    Normalized_Text         — full normalized text (used to reconstruct sentence boundaries)
    Sentences               — pre-split sentence strings (from SBD)
 
Writes:
    Targeted_Aspects — list of aspect dicts, added only when the record (or
                       comment) is subjective. Skipped records get an empty list.
 
Output format (one entry per entity × sentence match):
    {
        "Aspect_Name":        "VS Code",
        "Entity_Type":        "EDITOR",
        "Target_Sentence":    "I've always loved Cline but...",
        "Sentence_Word_Count": 14
    }
 
Fix: two-tier sentence interval search
---------------------------------------
The original code used a single str.find() to locate each sentence from
Sentences inside Normalized_Text. This produced the warning:
 
    "Could not locate sentence in Normalized_Text: 'Help ! !...'"
 
Root cause: SBD's _postprocess pass restores placeholder tokens (e.g.
SBDPROTECTVERVER0END → v1.0.3) which can leave extra or irregular
whitespace in the restored sentence. Normalized_Text (Stage 1 output)
always has whitespace collapsed to single spaces, so the strings diverge.
 
Fix: _build_sentence_intervals now runs two search tiers before giving up:
 
  Tier 1 — exact str.find() (original, fast, zero overhead when it works)
  Tier 2 — whitespace-normalised fallback: collapse all \s+ runs in the
            sentence to a single space, then search again. This matches
            sentences whose only difference from Normalized_Text is extra
            internal whitespace introduced by placeholder restoration.
 
The method now also returns the canonical sentence text (the version that
actually matched in Normalized_Text) alongside each interval so that
Target_Sentence in every aspect dict always reflects the real text.
 
Requires: no additional dependencies beyond the standard library.
"""
 
import json
import logging
import re
import regex
from pathlib import Path
from typing import Optional
 
logger = logging.getLogger(__name__)
 
# Pre-compiled once at module level — used in the Tier 2 fallback.
_WS = re.compile(r'\s+')
 
# Configuration
 
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
        self.aspect_entity_types: frozenset[str] = (
            aspect_entity_types if aspect_entity_types is not None
            else ASPECT_ENTITY_TYPES
        )

    def extract_record(self, record: dict) -> dict:
        record["Targeted_Aspects"] = self._extract(record)
        for comment in record.get("Comments") or []:
            comment["Targeted_Aspects"] = self._extract(comment)
        return record
 
    def extract_corpus(self, records: list[dict]) -> list[dict]:
        total = len(records)
        for i, record in enumerate(records, 1):
            try:
                self.extract_record(record)
            except Exception as e:
                logger.error(
                    f"AspectExtractor failed on record {record.get('ID', i)}: {e}"
                )
            if i % 500 == 0:
                logger.info(f"Aspect extraction: {i}/{total} records processed.")
        logger.info(f"Aspect extraction complete. {total} records processed.")
        return records
 
    # Core extraction 
    def _extract(self, container: dict) -> list[dict]:
        """
        Extract targeted aspects from a single record or comment dict.
 
        Returns [] immediately if the container is not subjective, has no
        NER_Tags, or has no Sentences.
        """
        if container.get("Subjectivity", "objective") != "subjective":
            return []
 
        ner_tags       : list[list] = container.get("NER_Tags") or []
        sentences      : list[str]  = container.get("Sentences") or []
        normalized_text: str        = container.get("Normalized_Text", "")
 
        if not ner_tags or not sentences:
            return []
 
        # Build [sent_start, sent_end) intervals + canonical sentence texts.
        sent_intervals, canonical_sentences = self._build_sentence_intervals(
            normalized_text, sentences
        )
 
        aspects: list[dict] = []
        seen: set[tuple]    = set()  # (entity_text_lower, sent_start)
 
        for entity in ner_tags:
            if len(entity) < 4:
                continue
 
            entity_text, entity_label, ent_start, ent_end = (
                entity[0], entity[1], entity[2], entity[3]
            )
 
            if entity_label not in self.aspect_entity_types:
                continue
 
            matched = self._match_sentence(
                ent_start, ent_end, sent_intervals, canonical_sentences
            )
            if matched is None:
                logger.debug(
                    f"Entity '{entity_text}' [{ent_start}:{ent_end}] "
                    "did not match any sentence interval — skipped."
                )
                continue
 
            # canonical_sent is the text that actually exists in Normalized_Text.
            canonical_sent, sent_start = matched
 
            key = (entity_text.lower(), sent_start)
            if key in seen:
                continue
            seen.add(key)
 
            aspects.append({
                "Aspect_Name":         entity_text,
                "Entity_Type":         entity_label,
                "Target_Sentence":     canonical_sent,
                "Sentence_Word_Count": len(canonical_sent.split()),
            })
 
        return aspects
 

    # Sentence interval reconstruction (two-tier search)
    def _build_sentence_intervals(
        self,
        normalized_text: str,
        sentences: list[str],
    ) -> tuple[list[tuple[int, int]], list[str]]:
        """
        Reconstruct [start, end) character intervals for each sentence inside
        Normalized_Text, scanning left-to-right to handle repeated sentences.
 
        Returns
        -------
        intervals         : list of (start, end) tuples, aligned with sentences.
        canonical_sentences : list of the sentence texts that actually matched
                             in Normalized_Text (may differ from Sentences[i]
                             in whitespace when the Tier 2 fallback was used).
 
        Search tiers
        ------------
        Tier 1 — exact str.find().  Zero overhead; works for the vast majority
                 of sentences where Stage 1 and SBD produce identical text.
 
        Tier 2 — whitespace-normalised fallback.  Collapses all \\s+ runs in
                 the sentence to a single space and searches again.  Catches
                 sentences where SBD's placeholder restoration introduced extra
                 internal whitespace (the root cause of the warning).
 
        If both tiers fail the sentence is skipped (sentinel interval appended,
        warning logged) and processing continues — no exception is raised.
        """
        intervals          : list[tuple[int, int]] = []
        canonical_sentences: list[str]             = []
        search_from        : int                   = 0
 
        for sentence in sentences:
            if not sentence:
                intervals.append((search_from, search_from))
                canonical_sentences.append(sentence)
                continue
 
            # ── Tier 1: exact match ───────────────────────────────────
            idx = normalized_text.find(sentence, search_from)
            if idx == -1:
                # Try from the beginning in case cursor overshot.
                idx = normalized_text.find(sentence)
 
            if idx != -1:
                end = idx + len(sentence)
                intervals.append((idx, end))
                canonical_sentences.append(sentence)
                search_from = end
                continue
 
            # Tier 2: whitespace-normalised fallback 
            # Remove all internal white space into single space cuz SBD leaves extra space
            sentence_norm = _WS.sub(' ', sentence).strip()
 
            if sentence_norm and sentence_norm != sentence:
                idx = normalized_text.find(sentence_norm, search_from)
                if idx == -1:
                    idx = normalized_text.find(sentence_norm)
 
                if idx != -1:
                    end = idx + len(sentence_norm)
                    intervals.append((idx, end))
                    canonical_sentences.append(sentence_norm)  # use matched text
                    search_from = end
                    logger.debug(
                        f"Sentence located via whitespace-normalised fallback: "
                        f"'{sentence_norm[:60]}'"
                    )
                    continue
            # Fuzzy search
            escaped_sent = re.escape(sentence_norm)
            
            # (e<=5) allows up to 5 insertions, deletions, or substitutions. 
            # BESTMATCH finds the substring with the fewest errors.
            fuzzy_pattern = regex.compile(f"({escaped_sent}){{e<=5}}", regex.BESTMATCH)
            
            match = fuzzy_pattern.search(normalized_text, search_from)
            if not match:
                # Try from the beginning if cursor overshot
                match = fuzzy_pattern.search(normalized_text)
                
            if match:
                idx = match.start()
                end = match.end()
                matched_text = match.group()
                
                intervals.append((idx, end))
                canonical_sentences.append(matched_text)
                search_from = end
                logger.warning(
                    f"Tier 3 Fuzzy Match Used: '{sentence[:40]}...' "
                    f"matched as '{matched_text[:40]}...'"
                )
                continue
 
            # If Both tiers failed
            # Append a sentinel so the lists stay aligned with Sentences[].
            # The entity containment check will simply never match this slot.
            logger.warning(
                f"Could not locate sentence in Normalized_Text: "
                f"'{sentence[:60]}'"
            )
            intervals.append((search_from, search_from))
            canonical_sentences.append(sentence)
 
        return intervals, canonical_sentences
 
    def _match_sentence(
        self,
        ent_start: int,
        ent_end: int,
        sent_intervals: list[tuple[int, int]],
        canonical_sentences: list[str],
    ) -> Optional[tuple[str, int]]:
        """
        Return (canonical_sentence_text, sentence_start) for the first
        sentence whose [sent_start, sent_end) interval strictly contains
        [ent_start, ent_end).
 
        Containment condition (half-open intervals, no off-by-one needed):
            ent_start >= sent_start  AND  ent_end <= sent_end
        """
        for (sent_start, sent_end), sent_text in zip(
            sent_intervals, canonical_sentences
        ):
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