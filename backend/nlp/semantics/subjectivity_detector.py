"""
subjectivity_detector.py
SC4021 Information Retrieval 2026 — Subjectivity Detection Module

Hybrid subjectivity detector: lexicon-based scoring runs first on every
sentence. Sentences whose scores fall in an uncertain middle band are
escalated to a transformer-based zero-shot classifier for a second opinion.

Pipeline position:
    MicrotextNormalizer → SBD → POSTagger → NERTagger → SubjectivityDetector

Requires:
    pip install transformers torch nltk vaderSentiment
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

Fields added to each record and comment:
    Subjectivity             — "subjective" | "objective"  (record-level)
    Subjectivity_Score       — float 0.0–1.0, higher = more subjective
    Subjectivity_Sentences   — list of per-sentence dicts (see below)

Per-sentence dict format:
    {
        "text"   : str,
        "label"  : "subjective" | "objective",
        "score"  : float,          # 0.0 = fully objective, 1.0 = fully subjective
        "method" : "lexicon" | "transformer"
    }

Hybrid logic (per sentence):
    lexicon_score >= UPPER_THRESHOLD  → subjective  (lexicon, no transformer call)
    lexicon_score <= LOWER_THRESHOLD  → objective   (lexicon, no transformer call)
    otherwise                         → transformer decides (expensive path)
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2] 
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import re
import logging
from transformers import pipeline
from pathlib import Path
from typing import Optional, TYPE_CHECKING

try:
    import torch
except ImportError:
    torch = None

if TYPE_CHECKING:
    from syntactics.sbd import SentenceBoundaryDisambiguator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lexicon constants
# ---------------------------------------------------------------------------

# Threshold band: scores outside [LOWER, UPPER] are decided by the lexicon.
# Scores inside the band are escalated to the transformer.
LOWER_THRESHOLD = 0.20
UPPER_THRESHOLD = 0.75


BYPASS_SUBJECTIVITY = True


# Domaiin Keywords 
DOMAIN_KEYWORDS = {"code", "python", "api", "copilot", "cursor", "developer", "bug"}

# First-person pronouns — strong subjectivity signal
FIRST_PERSON = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
}

# Opinion-bearing adverbs — indicate personal stance
OPINION_ADVERBS = {
    "honestly", "frankly", "personally", "truly", "seriously",
    "genuinely", "literally", "clearly", "obviously", "definitely",
    "certainly", "absolutely", "basically", "essentially", "arguably",
    "apparently", "admittedly", "surprisingly", "unfortunately", "thankfully",
}

# Hedging words — uncertainty = personal judgement = subjectivity
HEDGE_WORDS = {
    "maybe", "perhaps", "probably", "possibly", "seemingly",
    "apparently", "seems", "feel", "think", "believe", "guess",
    "suppose", "reckon", "expect", "doubt", "wonder", "feel", "feels", "felt",
}

# Intensifiers — amplify opinion signal
INTENSIFIERS = {
    "very", "really", "extremely", "incredibly", "insanely",
    "ridiculously", "super", "so", "soo", "quite", "pretty",
    "utterly", "totally", "completely", "highly", "massively",
}

# Sentiment-bearing adjectives and verbs common in tech opinion text
TECH_OPINION_ADJ_AND_VERBS = {
    # positive
    "amazing", "awesome", "brilliant", "fantastic", "excellent",
    "great", "good", "nice", "useful", "powerful", "fast", "smooth",
    "intuitive", "productive", "efficient", "impressive", "solid",
    "reliable", "helpful", "clean", "elegant", "simple", "better",
    "best", "favorite", "favourite",
    # negative
    "bad", "terrible", "awful", "horrible", "broken", "slow",
    "annoying", "frustrating", "useless", "buggy", "worse", "worst",
    "mediocre", "disappointing", "limited", "poor", "messy", "clunky",
    "bloated", "overrated", "unreliable", "confusing",

    # Positive Verbs
    "save", "automate", "speed", "boost", "streamline", "solve", 
    "recommend", "love", "enjoy",
    # Negative Verbs
    "crash", "break", "hallucinate", "freeze", "fail", "suck", 
    "waste", "ruin", "force", "complicate",

    # Emotional reactions — very common in tool reviews
    "disappointed", "disappointing", "surprised", "impressed", "shocked",
    "excited", "thrilled", "worried", "concerned", "satisfied", "unsatisfied",
    "pleased", "displeased", "annoyed", "frustrated", "delighted", "upset",
    "happy", "unhappy", "glad", "regret", "regretful", "skeptical",
    "nervous", "confident", "doubtful",

    # Evaluative verbs missing from your current list
    "prefer", "prefer", "hate", "dislike", "like", "love", "miss",
    "appreciate", "enjoy", "avoid", "trust", "distrust", "depend", "rely",
    "struggle", "manage", "succeed", "fail",

    # Tech-specific reactions
    "switch", "switched", "migrate", "migrated", "abandon", "abandoned",
    "replaced", "dropped", "adopted", "tried", "tested",
}



# Weights for each lexicon signal (contribution to subjectivity score)
SIGNAL_WEIGHTS = {
    "mpqa":          0.20,   # MPQA strongsubj/weaksubj lexicon
    "vader":         0.20,   # VADER compound score absolute value
    "first_person":  0.18,   # presence of I/me/my etc.
    "opinion_adverb":0.10,   # honestly, frankly, personally...
    "hedge":         0.08,   # maybe, seems, think...
    "intensifier":   0.07,   # very, really, extremely...
    "tech_opinion":  0.10,   # domain-specific opinion adjectives/verbs
    "emoticon":      0.04,   # [smiley face] tokens from normalizer
    "exclamation":   0.03,   # ! in the sentence
}


DEFAULT_TRANSFORMER = "GroNLP/mdebertav3-subjectivity-english"


# ---------------------------------------------------------------------------
# SubjectivityDetector
# ---------------------------------------------------------------------------

class SubjectivityDetector:
    """
    Hybrid subjectivity detection for the SC4021 pipeline.

    Usage
    -----
    detector = SubjectivityDetector()
    record = detector.detect_record(record)

    For a full corpus:
    records = detector.detect_corpus(records)
    """

    def __init__(
        self,
        lower_threshold: float = LOWER_THRESHOLD,
        upper_threshold: float = UPPER_THRESHOLD,
        transformer_model: str = DEFAULT_TRANSFORMER,
        use_transformer: bool = True,
        mpqa_path: str = "../data/lexicons/mpqa_subjclues.tff",
    ):
        """
        Parameters
        ----------
        lower_threshold : lexicon scores at or below this → objective (no transformer)
        upper_threshold : lexicon scores at or above this → subjective (no transformer)
        transformer_model : HuggingFace model name for zero-shot classification
        use_transformer : set False to run lexicon-only (faster, for testing)
        mpqa_path : path to the MPQA subjectivity lexicon file
        """
        self.lower = lower_threshold
        self.upper = upper_threshold
        self.use_transformer = use_transformer

        # --- MPQA Lexicon ---
        mpqa_file = self._resolve_path(mpqa_path)
        self._mpqa = self._load_mpqa_lexicon(mpqa_file)

        # --- VADER Sentiment Analyzer ---
        self._vader = None
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer loaded.")
        except ImportError:
            logger.warning(
                "vaderSentiment not installed — VADER signal disabled. "
                "Install with: pip install vaderSentiment"
            )

        # --- HuggingFace transformer (lazy loaded on first uncertain sentence) ---
        self._classifier = None
        self._transformer_model = transformer_model
        if use_transformer:
            logger.info(
                f"Transformer '{transformer_model}' will be loaded on first "
                "uncertain sentence (lazy init)."
            )
        else:
            logger.info("Transformer disabled — running lexicon-only mode.")

    def _is_relevant(self, text: str, pos_tags: list[list]) -> bool:
        """
        Phase A: Topic/Relevance Check.
        Returns True if the text is about AI/Coding, False if it is off-topic.
        """
        # Flatten the POS tags to easily search all lemmas in the text
        all_lemmas = {lemma.lower() for sentence in pos_tags for _, _, lemma in sentence}
        
        # If there is no overlap between the text's lemmas and your keywords, it's irrelevant.
        if not all_lemmas.intersection(DOMAIN_KEYWORDS):
            return False
            
        # You can add more complex context/proximity checks here later!
        return True

    def detect_record(self, record: dict) -> dict:
        """
        Detect subjectivity for a single record and all its comments.

        Requires 'Normalized_Text' (from MicrotextNormalizer).
        POS_Tags (from POSTagger) are used if present to improve scoring.

        Adds: Subjectivity, Subjectivity_Score, Subjectivity_Sentences.
        Returns the modified record.
        """

        if BYPASS_SUBJECTIVITY:
            record["Subjectivity"] = "subjective"
            record["Subjectivity_Score"] = 1.0
            record["Subjectivity_Sentences"] = []
            for comment in (record.get("Comments") or []):
                comment["Subjectivity"] = "subjective"
                comment["Subjectivity_Score"] = 1.0
                comment["Subjectivity_Sentences"] = []
            return record
    
        text = record.get("Normalized_Text", "")
        pos_tags = record.get("POS_Tags", [])
        sentences = record.get("Sentences", [])   # ← read pre-split sentences

        if not text:
            logger.warning(f"Record {record.get('ID', '?')} has no Normalized_Text.")
            self._write_empty(record)
            return record
        
        if not self._is_relevant(text, pos_tags):
            record["Subjectivity"] = "Irrelevant"
            record["Subjectivity_Score"] = None
            record["Subjectivity_Sentences"] = []
        else:
            # Existing subjective/objective logic runs ONLY if relevant
            sentence_results = self._score_sentences(sentences, pos_tags)
            record["Subjectivity_Sentences"] = sentence_results
            record["Subjectivity"], record["Subjectivity_Score"] = self._aggregate(sentence_results)

        if record["Subjectivity"] != "Irrelevant": 
            post_context = (record.get("Title") or "")[:200]
            for comment in record.get("Comments") or []:
                c_text = comment.get("Normalized_Text", "")
                c_pos = comment.get("POS_Tags", [])
                c_sentences = comment.get("Sentences", [])   # ← read pre-split sentences

                if not c_text:
                    self._write_empty(comment)
                    continue

                c_results = self._score_sentences(c_sentences, c_pos, parent_context=post_context)
                comment["Subjectivity_Sentences"] = c_results
                comment["Subjectivity"], comment["Subjectivity_Score"] = (
                    self._aggregate(c_results)
                )
        else: 
            for comment in record.get("Comments") or []:
                comment["Subjectivity"] = "Irrelevant"
                comment["Subjectivity_Score"] = None
                comment["Subjectivity_Sentences"] = []

        return record

    def detect_corpus(self, records: list[dict]) -> list[dict]:
        """Detect subjectivity for an entire list of records."""

        if BYPASS_SUBJECTIVITY:
            for record in records:
                record["Subjectivity"] = "subjective"
                record["Subjectivity_Score"] = 1.0
                record["Subjectivity_Sentences"] = []
                for comment in (record.get("Comments") or []):
                    comment["Subjectivity"] = "subjective"
                    comment["Subjectivity_Score"] = 1.0
                    comment["Subjectivity_Sentences"] = []
            logger.info("SubjectivityDetector: BYPASS active — placeholder values written, model not loaded.")
            return records


        total = len(records)
        for i, record in enumerate(records, 1):
            try:
                self.detect_record(record)
            except Exception as e:
                logger.error(
                    f"Failed on record {record.get('ID', i)}: {e}"
                )
            if i % 500 == 0:
                logger.info(f"Processed {i}/{total} records...")
        logger.info(f"Subjectivity detection complete. {total} records processed.")
        return records

    # Core detection

    def _score_sentences(
        self,
        sentences: list[str],
        pos_tags: list[list],
        parent_context: str = "",
    ) -> list[dict]:
        results = [None] * len(sentences)
        transformer_queue = []  # (sentence_idx, sentence, context)

        # --- Pass 1: Lexicon pass — resolve what we can, queue the rest ---
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                results[i] = {
                    "text": sentence, "label": "objective",
                    "score": 0.0, "method": "lexicon"
                }
                continue

            sentence_pos = pos_tags[i] if i < len(pos_tags) else []
            pos_lookup   = self._build_pos_lookup(sentence_pos)
            lexicon_score = self._lexicon_score(sentence, pos_lookup)

            word_count = len(sentence.split())
            in_uncertain_band = self.lower < lexicon_score < self.upper
            force_transformer = (
                self.use_transformer
                and word_count <= 10
                and lexicon_score > 0.05
                and in_uncertain_band
            )

            if not force_transformer and lexicon_score >= self.upper:
                results[i] = {
                    "text": sentence, "label": "subjective",
                    "score": round(lexicon_score, 4), "method": "lexicon"
                }
            elif not force_transformer and lexicon_score <= self.lower:
                results[i] = {
                    "text": sentence, "label": "objective",
                    "score": round(lexicon_score, 4), "method": "lexicon"
                }
            else:
                # Queue for batched transformer inference
                transformer_queue.append((i, sentence, lexicon_score, parent_context))

        # --- Pass 2: Batched transformer inference ---
        if transformer_queue and self.use_transformer:
            self._load_transformer()

            # Build cleaned input texts
            inputs = []
            for (_, sentence, lexicon_score, context) in transformer_queue:
                clean = re.sub(r"<CODE>|\[[^\]]+\]", "", sentence).strip()
                if context and len(clean.split()) <= 10:
                    inputs.append(f"{context[:200]} | {clean}")
                else:
                    inputs.append(clean or ".")

            # Single batched call — this is the key change
            try:
                batch_results = self._classifier(
                    inputs,
                    batch_size=32,      # tune based on your RAM
                    truncation=True,
                    max_length=512,
                )
            except Exception as e:
                logger.warning(f"Batch transformer failed: {e} — falling back to neutral")
                batch_results = [{"label": "OBJ", "score": 0.5}] * len(inputs)

            # Write batch results back
            for (sent_idx, sentence, lexicon_score, _), model_out in zip(
                transformer_queue, batch_results
            ):
                raw_label = model_out["label"]
                t_score   = model_out["score"]

                if any(k in raw_label.upper() for k in ("SUBJ",)):
                    t_label = "subjective"
                else:
                    t_label = "objective"
                    t_score = 1.0 - t_score   # convert OBJ confidence to SUBJ probability

                # Neutral flip for low-confidence objective predictions
                if t_label == "objective" and model_out["score"] < 0.65:
                    has_first_person = bool(re.search(
                        r'\b(i|my|me|we|our)\b', sentence, re.IGNORECASE
                    ))
                    has_stance = bool(re.search(
                        r'\b(i think|i feel|imo|imho|ngl|tbh|never|always)\b',
                        sentence, re.IGNORECASE
                    ))
                    if has_first_person and has_stance:
                        t_label = "subjective"
                        t_score = 1.0 - model_out["score"]

                blended_score = round(0.4 * lexicon_score + 0.6 * t_score, 4)
                label = "subjective" if blended_score >= 0.5 else "objective"

                results[sent_idx] = {
                    "text":   sentence,
                    "label":  label,
                    "score":  blended_score,
                    "method": "transformer",
                }

        return [r for r in results if r is not None]



    def _aggregate(self, sentence_results: list[dict]) -> tuple[str, float]:
        """
        Aggregate sentence-level results into a record-level label.

        Strategy: blend of mean and max scores.
        - Mean captures overall tone
        - Max captures the strongest opinion signal in any single sentence
        A record is subjective if the blended score exceeds 0.45.
        """
        if not sentence_results:
            return "objective", 0.0

        scores = [s["score"] for s in sentence_results]
        mean_score = sum(scores) / len(scores)
        max_score = max(scores)

        # 40% mean, 60% max — rewards posts with at least one strong opinion
        blended = 0.4 * mean_score + 0.6 * max_score

        # Slightly lower decision boundary than 0.5 since we're blending
        label = "subjective" if blended >= 0.42 else "objective"
        return label, round(blended, 4)

    # ------------------------------------------------------------------
    # Lexicon scoring
    # ------------------------------------------------------------------

    def _lexicon_score(self, sentence, pos_lookup):
        tokens = [t.strip('.,!?()[]"').lower() for t in sentence.split()]
        token_set = set(tokens)

        # Base score from VADER — this is the foundation, not just one signal
        if self._vader:
            clean = re.sub(r"<CODE>|\[[^\]]+\]", "", sentence).strip()
            vader_compound = abs(self._vader.polarity_scores(clean)["compound"])
        else:
            vader_compound = 0.5   # unknown → push to transformer

        # Heuristic adjustment — each signal shifts the base score
        adjustment = 0.0
        if token_set & FIRST_PERSON:          adjustment += 0.12
        if token_set & OPINION_ADVERBS:       adjustment += 0.08
        if token_set & HEDGE_WORDS:           adjustment += 0.06
        if token_set & INTENSIFIERS:          adjustment += 0.05
        if token_set & TECH_OPINION_ADJ_AND_VERBS:  adjustment += 0.10
        if self._mpqa:
            strong = sum(1 for t in tokens if self._mpqa.get(t, {}).get("type") == "strongsubj")
            weak   = sum(1 for t in tokens if self._mpqa.get(t, {}).get("type") == "weaksubj")
            adjustment += min((strong * 0.08 + weak * 0.04), 0.15)
        if re.search(r"\[[^\]]+\]", sentence):  adjustment += 0.04   # emoticon
        if "!" in sentence:                     adjustment += 0.03

        score = vader_compound + adjustment
        return min(max(score, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Transformer scoring
    # ------------------------------------------------------------------

    def _load_transformer(self) -> None:
        if self._classifier is not None:
            return
        try:
            logger.info(f"Loading transformer '{self._transformer_model}'...")

            # Use CUDA when available; device=-1 forces CPU in transformers.
            device = -1
            device_desc = "CPU"
            if torch is None:
                logger.warning(
                    "PyTorch is not installed; transformer will run on CPU. "
                    "Install torch with CUDA support to use GPU."
                )
            elif torch.cuda.is_available():
                device = 0
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    device_desc = f"GPU (cuda:0, {gpu_name})"
                except Exception:
                    device_desc = "GPU (cuda:0)"
            else:
                logger.info("CUDA not available; transformer will run on CPU.")

            self._classifier = pipeline(
                "text-classification",
                model=self._transformer_model,
                device=device,
            )
            logger.info(f"Transformer loaded on {device_desc}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load transformer: {e}")
            

    def _transformer_score(self, sentence: str, context: str = "") -> tuple[str, float]:
        self._load_transformer()

        clean = re.sub(r"<CODE>|\[[^\]]+\]", "", sentence).strip()
        if not clean:
            return "objective", 0.0

        # Prepend context as a premise only for short sentences
        if context and len(clean.split()) <= 10:
            input_text = f"{context[:200]} | {clean}"  # cap context length
        else:
            input_text = clean

        classifier = self._classifier
        if classifier is None:
            raise RuntimeError("Transformer classifier is not initialized")

        result = classifier(input_text)[0]
        label = "subjective" if result["label"] == "SUBJ" else "objective"
        score = result["score"] if result["label"] == "SUBJ" else 1 - result["score"]
        return label, float(score)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------


    def _build_pos_lookup(
        self, pos_tags: list[list]
    ) -> dict[str, tuple[str, str, str]]:
        """
        Build a dict from token surface form → (surface, pos, lemma)
        for quick lookup during lexicon scoring.
        """
        return {
            token.lower(): (token, pos, lemma)
            for token, pos, lemma in pos_tags
        }

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        """Resolve file path from cwd first, then relative to this module."""
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        return (Path(__file__).resolve().parent / candidate).resolve()

    def _load_mpqa_lexicon(self, path: str | Path) -> dict[str, dict]:
        """
        Load MPQA subjectivity lexicon into a dict keyed by word.
        Each entry: { "type": "strongsubj"|"weaksubj", "pos": str, "polarity": str }
        Returns empty dict if file not found (degrades gracefully).
        """
        path = Path(path)
        lexicon: dict[str, dict] = {}
        if not path.exists():
            logger.warning(
                f"MPQA lexicon not found at '{path}'. "
                "MPQA signal will be disabled."
            )
            return lexicon
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = dict(
                    item.split("=") for item in line.strip().split()
                    if "=" in item
                )
                word = parts.get("word1", "")
                if word:
                    lexicon[word.lower()] = {
                        "type":     parts.get("type", "weaksubj"),
                        "pos":      parts.get("pos1", "anypos"),
                        "polarity": parts.get("priorpolarity", "neutral"),
                    }
        logger.info(f"MPQA lexicon loaded: {len(lexicon)} entries from {path}")
        return lexicon

    @staticmethod
    def _write_empty(container: dict) -> None:
        """Write empty subjectivity fields to a record or comment."""
        container["Subjectivity"] = "objective"
        container["Subjectivity_Score"] = 0.0
        container["Subjectivity_Sentences"] = []



# ---------------------------------------------------------------------------
# Example usage / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulates a record after MicrotextNormalizer + POSTagger + NERTagger.
    sample_record = {
        "ID": "r_001",
        "Source": "Reddit",
        "Normalized_Text": (
            "GitHub Copilot was released by Microsoft in 2021. "
            "Honestly it has made me so much more productive. "
            "I think it is better than tabnine for Python development. "
            "The autocomplete feature suggests entire functions which is amazing. "
            "Some people say it produces buggy code sometimes."
        ),
        "Sentences": [
            "GitHub Copilot was released by Microsoft in 2021.",
            "Honestly it has made me so much more productive.",
            "I think it is better than tabnine for Python development.",
            "The autocomplete feature suggests entire functions which is amazing.",
            "Some people say it produces buggy code sometimes.",
        ],
        # Sentence-aligned: POS_Tags[i] ↔ Sentences[i], content words only
        "POS_Tags": [
            [["GitHub", "PROPN", "GitHub"], ["Copilot", "PROPN", "Copilot"],
             ["released", "VERB", "release"], ["Microsoft", "PROPN", "Microsoft"]],
            [["Honestly", "ADV", "honestly"], ["made", "VERB", "make"],
             ["productive", "ADJ", "productive"]],
            [["think", "VERB", "think"], ["better", "ADJ", "well"],
             ["tabnine", "NOUN", "tabnine"], ["Python", "PROPN", "Python"],
             ["development", "NOUN", "development"]],
            [["autocomplete", "NOUN", "autocomplete"], ["feature", "NOUN", "feature"],
             ["suggests", "VERB", "suggest"], ["functions", "NOUN", "function"],
             ["amazing", "ADJ", "amazing"]],
            [["people", "NOUN", "people"], ["say", "VERB", "say"],
             ["produces", "VERB", "produce"], ["buggy", "ADJ", "buggy"],
             ["code", "NOUN", "code"]],
        ],
        "Comments": [
            {
                "comment_id": "c_001",
                "Normalized_Text": (
                    "I completely agree. "
                    "The context window is 8192 tokens. "
                    "It feels incredibly intuitive to use [smiley face]."
                ),
                "Sentences": [
                    "I completely agree.",
                    "The context window is 8192 tokens.",
                    "It feels incredibly intuitive to use [smiley face].",
                ],
                "POS_Tags": [
                    [["completely", "ADV", "completely"], ["agree", "VERB", "agree"]],
                    [["context", "NOUN", "context"], ["window", "NOUN", "window"],
                     ["8192", "NUM", "8192"], ["tokens", "NOUN", "token"]],
                    [["feels", "VERB", "feel"], ["incredibly", "ADV", "incredibly"],
                     ["intuitive", "ADJ", "intuitive"],
                     ["[smiley face]", "EMOTICON", "[smiley face]"]],
                ],
            }
        ],
    }
    
    sample_record_irrelevant = {
        "ID": "r_002",
        "Source": "Reddit",
        "Normalized_Text": (
            "I went to the new Italian place downtown yesterday. "
            "Honestly the pizza crust was perfectly crispy and the cheese was amazing. "
            "I highly recommend trying their garlic bread. "
            "It cost about 20 dollars which is a great deal."
        ),
        "Sentences": [
            "I went to the new Italian place downtown yesterday.",
            "Honestly the pizza crust was perfectly crispy and the cheese was amazing.",
            "I highly recommend trying their garlic bread.",
            "It cost about 20 dollars which is a great deal.",
        ],
        # Sentence-aligned: POS_Tags[i] ↔ Sentences[i], content words only
        "POS_Tags": [
            [["went", "VERB", "go"], ["new", "ADJ", "new"], ["Italian", "ADJ", "italian"],
             ["place", "NOUN", "place"], ["downtown", "ADV", "downtown"], ["yesterday", "NOUN", "yesterday"]],
            [["Honestly", "ADV", "honestly"], ["pizza", "NOUN", "pizza"], ["crust", "NOUN", "crust"],
             ["perfectly", "ADV", "perfectly"], ["crispy", "ADJ", "crispy"], ["cheese", "NOUN", "cheese"],
             ["amazing", "ADJ", "amazing"]],
            [["highly", "ADV", "highly"], ["recommend", "VERB", "recommend"],
             ["trying", "VERB", "try"], ["garlic", "NOUN", "garlic"], ["bread", "NOUN", "bread"]],
            [["cost", "VERB", "cost"], ["20", "NUM", "20"], ["dollars", "NOUN", "dollar"],
             ["great", "ADJ", "great"], ["deal", "NOUN", "deal"]],
        ],
        "Comments": [
            {
                "comment_id": "c_002",
                "Normalized_Text": (
                    "I completely agree. "
                    "The tomato sauce tasted super fresh. "
                    "Best meal I have had in weeks!"
                ),
                "Sentences": [
                    "I completely agree.",
                    "The tomato sauce tasted super fresh.",
                    "Best meal I have had in weeks!",
                ],
                "POS_Tags": [
                    [["completely", "ADV", "completely"], ["agree", "VERB", "agree"]],
                    [["tomato", "NOUN", "tomato"], ["sauce", "NOUN", "sauce"],
                     ["tasted", "VERB", "taste"], ["super", "ADV", "super"], ["fresh", "ADJ", "fresh"]],
                    [["Best", "ADJ", "good"], ["meal", "NOUN", "meal"], ["had", "VERB", "have"],
                     ["weeks", "NOUN", "week"]],
                ],
            }
        ],
    }

    detector = SubjectivityDetector(use_transformer=True)
    result = detector.detect_record(sample_record_irrelevant)

    print(f"\n=== Record-level ===")
    print(f"  Label : {result['Subjectivity']}")
    print(f"  Score : {result['Subjectivity_Score']}")

    print(f"\n=== Sentence-level (Post) ===")
    print(f"  {'Sentence':<60} {'Label':<12} {'Score':>6} {'Method'}")
    print(f"  {'-'*60} {'-'*12} {'-'*6} {'-'*11}")
    for s in result["Subjectivity_Sentences"]:
        preview = s["text"][:57] + "..." if len(s["text"]) > 60 else s["text"]
        print(f"  {preview:<60} {s['label']:<12} {s['score']:>6.3f} {s['method']}")

    print(f"\n=== Sentence-level (Comment) ===")
    comment = result["Comments"][0]
    print(f"  Label : {comment['Subjectivity']}")
    print(f"  Score : {comment['Subjectivity_Score']}")
    for s in comment["Subjectivity_Sentences"]:
        preview = s["text"][:57] + "..." if len(s["text"]) > 60 else s["text"]
        print(f"  {preview:<60} {s['label']:<12} {s['score']:>6.3f} {s['method']}")