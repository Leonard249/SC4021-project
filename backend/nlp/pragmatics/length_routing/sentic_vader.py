"""
sentic_vader.py
SC4021 Information Retrieval 2026 — Short-Text Polarity Classifier

Pipeline position:
    ... → SarcasmDetector → PolarityEnsemble → SenticVaderClassifier ← here
                                                (dispatched for Target_Sentences < 60 words)

Input
-----
    target_sentence : str       — Target_Sentence from a Targeted_Aspects entry
    pos_tags        : list      — flat list of [token, POS, lemma] triples
                                  sourced from the parent record's POS_Tags
                                  (used for SenticNet lemma lookup)

Output (in-memory dict, not written to record — Ensemble owns record writes)
------
    {
        "Label":        "positive" | "negative" | "neutral",
        "Score":         float,   # 0.0–1.0 on positive axis
        "Confidence":    float,   # 0.0–1.0, distance from 0.5 midpoint × 2
        "Routing_Path":  "short",
        "Classifier":    "sentic_vader"
    }

Fusion strategy
---------------
VADER compound score and SenticNet mean polarity_value are each normalised
to [0, 1] then fused with a weighted average:

    vader_norm = (compound + 1) / 2          maps −1…+1 → 0…1
    sn_norm    = (mean_polarity + 1) / 2      maps −1…+1 → 0…1

    final_score = VADER_WEIGHT * vader_norm + SENTICNET_WEIGHT * sn_norm

    VADER_WEIGHT     = 0.55   (preferred — handles negation and punctuation)
    SENTICNET_WEIGHT = 0.45

If SenticNet is unavailable or returns no hits, the classifier falls back to
VADER alone (effective weight 1.0).

Domain negative lexicon boost
------------------------------
VADER is known to under-score negativity in AI/coding discourse — phrases like
"hallucinating garbage" or "completely useless" are scored near-neutral.
A domain-specific negative lexicon is applied directly to the target sentence
(not the parent POS tags). Each match shifts the fused score toward negative by
DOMAIN_NEG_BOOST per match, capped at DOMAIN_NEG_CAP total shift.

Label thresholds (asymmetric — negative bar is intentionally lower):
    score >= 0.55  → "positive"
    score <= 0.48  → "negative"   ← lowered from 0.45 to catch hedged negativity
    otherwise      → "neutral"

Requires:
    pip install vaderSentiment senticnet
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns are matched case-insensitively against the *cleaned* target sentence.
# Use word-boundary anchors (\b) where a substring match would cause false fires.
DOMAIN_NEGATIVE_PATTERNS: list[str] = [
    # Reliability / correctness failures
    r"\bhallucinate[sd]?\b", r"\bhallucination[s]?\b",
    r"\bwrong(?:ly)?\b",     r"\bincorrect(?:ly)?\b",
    r"\bbroken\b",           r"\bregress(?:ed|ion|ions)?\b",
    r"\bdegrades?\b",        r"\bdegraded\b",
    r"\bbug(?:gy|s)?\b",     r"\bcrash(?:es|ed|ing)?\b",
    r"\bcorrupt(?:s|ed|ion)?\b",

    # Productivity / workflow damage
    r"\bslows?\s+(?:me\s+)?down\b", r"\bkill(?:s|ed|ing)?\s+(?:my\s+)?(?:workflow|productivity)\b",
    r"\bwaste[sd]?\s+(?:my\s+)?time\b", r"\bfrustrat(?:ing|ed|es?)\b",
    r"\bpointless\b",        r"\buseless\b",
    r"\bunreliable\b",       r"\bunstable\b",
    r"\bnot\s+worth\b",      r"\bwaste\s+of\b",

    # Trust / quality concerns
    r"\bcan(?:'t|not)\s+trust\b", r"\bdon(?:'t|ot)\s+trust\b",
    r"\bmisleading\b",       r"\binaccurate\b",
    r"\boutdated\b",         r"\bobsolete\b",
    r"\bdisappoint(?:ing|ed|s)?\b",

    # Abandonment / rejection signals
    r"\bswitched?\s+(?:back|away|to)\b",
    r"\bstopped?\s+using\b", r"\buninstall(?:ed|ing)?\b",
    r"\bremoved?\s+(?:it|the\s+\w+)\b",
    r"\bgave?\s+up\b",       r"\bno\s+longer\s+use\b",

    # Comparative inferiority
    r"\bworse\s+than\b",     r"\bfar\s+worse\b",
    r"\bmuch\s+worse\b",     r"\bfar\s+inferior\b",

    # Strong negative intensifiers common in tech discourse
    r"\bcompletely\s+(?:useless|broken|wrong|missed)\b",
    r"\babsolutely\s+(?:useless|broken|terrible|awful)\b",
    r"\btotal(?:ly)?\s+(?:garbage|trash|mess|failure)\b",
    r"\b(?:garbage|trash|junk)\b",
    r"\bterrible\b",         r"\bawful\b",     r"\bhorrible\b",
    r"\bdreadful\b",         r"\babysmal\b",
]

# Pre-compile all domain negative patterns once at import time for speed.
_DOMAIN_NEG_RE: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in DOMAIN_NEGATIVE_PATTERNS
]

# ---------------------------------------------------------------------------
# Weights & thresholds
# ---------------------------------------------------------------------------

VADER_WEIGHT      = 0.55
SENTICNET_WEIGHT  = 0.45
POSITIVE_THRESHOLD = 0.55
NEGATIVE_THRESHOLD = 0.48   # Lowered from 0.45 — catches hedged / understated negativity

# ---------------------------------------------------------------------------
# Domain-specific negative lexicon for AI/coding discourse
# ---------------------------------------------------------------------------
# VADER scores many of these terms near-neutral because they are rare in
# its training corpus (movie/product reviews). Each regex pattern that fires
# shifts the fused score toward negative by DOMAIN_NEG_BOOST, capped at
# DOMAIN_NEG_CAP total shift so a single very negative sentence can't push
# the score below 0 on its own.

DOMAIN_NEG_BOOST = 0.06   # score penalty per matched pattern
DOMAIN_NEG_CAP   = 0.24   # maximum total penalty (= 4 pattern hits)

# ---------------------------------------------------------------------------
# SenticVaderClassifier
# ---------------------------------------------------------------------------

class SenticVaderClassifier:
    """
    Short-text polarity classifier combining VADER and SenticNet.

    Called by PolarityEnsemble for Target_Sentences with fewer than 60 words.

    Usage
    -----
    classifier = SenticVaderClassifier()
    result = classifier.classify(target_sentence, pos_tags)
    """

    def __init__(self) -> None:
        self._vader     = None
        self._senticnet = None
        self._load_vader()
        self._load_senticnet()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        target_sentence: str,
        pos_tags: Optional[list[list]] = None,
    ) -> dict:
        """
        Classify the polarity of a short Target_Sentence.

        Parameters
        ----------
        target_sentence : Target_Sentence from a Targeted_Aspects entry.
        pos_tags        : flat list of [token, POS, lemma] triples from the
                          parent record's POS_Tags (used for SenticNet lookup).
                          Pass [] or None when unavailable.

        Returns
        -------
        {
            "Label":       "positive" | "negative" | "neutral",
            "Score":        float,   # 0.0–1.0
            "Confidence":   float,   # 0.0–1.0
            "Routing_Path": "short",
            "Classifier":   "sentic_vader"
        }
        """
        clean = re.sub(r"<CODE>|\[[^\]]+\]", "", target_sentence).strip()
        if not clean:
            return self._neutral_result()

        vader_norm = self._vader_score(clean)
        sn_norm    = self._senticnet_score(pos_tags or [])

        if sn_norm is None:
            # SenticNet unavailable / no hits — VADER only.
            final = vader_norm
        else:
            final = VADER_WEIGHT * vader_norm + SENTICNET_WEIGHT * sn_norm

        # Apply domain-specific negative boost AFTER fusion so it can correct
        # VADER's well-known positive bias on AI/coding vocabulary.
        domain_penalty = self._domain_neg_penalty(clean)
        final = max(0.0, final - domain_penalty)

        label      = self._label(final)
        confidence = round(abs(final - 0.5) * 2, 4)

        return {
            "Label":        label,
            "Score":        round(final, 4),
            "Confidence":   confidence,
            "Routing_Path": "short",
            "Classifier":   "sentic_vader",
        }

    # ------------------------------------------------------------------
    # Domain negative lexicon
    # ------------------------------------------------------------------

    @staticmethod
    def _domain_neg_penalty(text: str) -> float:
        """
        Count how many domain-negative patterns fire on *text* and return a
        score penalty in [0, DOMAIN_NEG_CAP].

        Each pattern match contributes DOMAIN_NEG_BOOST. The total is capped at
        DOMAIN_NEG_CAP so that even a very negative sentence cannot push the
        fused score below 0.0.
        """
        hits = sum(1 for pattern in _DOMAIN_NEG_RE if pattern.search(text))
        return min(hits * DOMAIN_NEG_BOOST, DOMAIN_NEG_CAP)

    # ------------------------------------------------------------------
    # VADER
    # ------------------------------------------------------------------

    def _vader_score(self, text: str) -> float:
        if self._vader is None:
            return 0.5
        compound = self._vader.polarity_scores(text)["compound"]
        return (compound + 1.0) / 2.0

    def _load_vader(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.info("SenticVaderClassifier: VADER loaded.")
        except ImportError:
            logger.warning(
                "vaderSentiment not installed — VADER signal disabled. "
                "Install with: pip install vaderSentiment"
            )

    # ------------------------------------------------------------------
    # SenticNet
    # ------------------------------------------------------------------

    def _senticnet_score(self, pos_tags: list[list]) -> Optional[float]:
        """
        Look up each lemma in SenticNet; return mean polarity_value
        normalised to [0, 1]. Returns None if no hits or SenticNet absent.
        """
        if self._senticnet is None:
            return None

        polarities: list[float] = []
        for entry in pos_tags:
            if len(entry) < 3:
                continue
            lemma = entry[2].lower().strip()
            if not lemma:
                continue
            try:
                concept = self._senticnet.concept(lemma)
                pv = float(concept.get("polarity_value", 0.0))
                polarities.append(pv)
            except Exception:
                continue  # lemma not in SenticNet

        if not polarities:
            return None

        mean_pv = sum(polarities) / len(polarities)
        return (mean_pv + 1.0) / 2.0

    def _load_senticnet(self) -> None:
        try:
            from senticnet.senticnet import SenticNet
            self._senticnet = SenticNet()
            logger.info("SenticVaderClassifier: SenticNet loaded.")
        except ImportError:
            logger.warning(
                "senticnet not installed — SenticNet signal disabled. "
                "Install with: pip install senticnet"
            )
        except Exception as e:
            logger.warning(f"SenticVaderClassifier: SenticNet failed to load: {e}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _label(score: float) -> str:
        if score >= POSITIVE_THRESHOLD:
            return "positive"
        if score <= NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"

    @staticmethod
    def _neutral_result() -> dict:
        return {
            "Label":        "neutral",
            "Score":        0.5,
            "Confidence":   0.0,
            "Routing_Path": "short",
            "Classifier":   "sentic_vader",
        }


# ---------------------------------------------------------------------------
# Utility — flatten sentence-aligned POS_Tags into a single list
# ---------------------------------------------------------------------------

def flatten_pos_tags(pos_tags: list[list[list]]) -> list[list]:
    """
    Convert sentence-aligned POS_Tags (list of lists of triples) into a flat
    list of [token, POS, lemma] triples for SenticNet lemma lookup.
    """
    flat: list[list] = []
    for sentence_tags in pos_tags:
        if isinstance(sentence_tags, list):
            flat.extend(sentence_tags)
    return flat


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    # Updated to the exact path requested
    dataset_path = Path('../../../../data/my_test/sentic_vader_input.json')

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    records: list[dict] = data if isinstance(data, list) else [data]

    classifier = SenticVaderClassifier()

    for record in records:
        flat_pos = flatten_pos_tags(record.get("POS_Tags") or [])
        for asp in record.get("Targeted_Aspects") or []:
            sent = asp.get("Target_Sentence", "")
            wc   = asp.get("Sentence_Word_Count", len(sent.split()))
            
            if wc >= 60:
                continue  # would be handled by transformer
            
            # Fetch the in-memory dictionary
            result_to_print = classifier.classify(sent, flat_pos)
            
            # Print exactly the dictionary output formatted with indent=4
            print(json.dumps(result_to_print, indent=4))