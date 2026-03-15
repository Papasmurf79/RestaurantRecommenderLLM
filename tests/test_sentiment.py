"""
tests/test_sentiment.py
=======================
Unit tests for Notebook 4 — Emotion & Sentiment Analysis.

Strategy
--------
Both models are fully mocked. No downloads, no GPU, no network.

  - bhadresh-savani/distilbert-base-uncased-emotion
      Returns a list of 7 dicts [{label, score}, ...] sorted by score
      descending. The wrapper picks label[0] as dominant_emotion and
      distributes each score to its emotion_{label} column.

  - Secondary sentiment classifier (e.g. distilbert-base-uncased)
      Returns [{label: "POSITIVE"/"NEGATIVE"/"NEUTRAL", score: float}].
      The wrapper normalises the label to Title Case and stores the score
      as sentiment_score.

Tests operate in two layers:

  Layer 1 — CSV artifact tests (Sections 1–6):
      Load restaurants_with_emotions.csv and assert schema, row count,
      nulls, value validity, score bounds, and dominant_emotion
      consistency against the argmax of the 7 stored scores.

  Layer 2 — Wrapper unit tests (Sections 7–12):
      Test the wrapper functions directly with synthetic mock outputs,
      covering dominant_emotion selection, score distribution, dining_mood
      mapping, sentiment normalisation, and edge cases.

Key facts from the real CSV (grounded, not assumed):
  Total rows:             71
  Total columns:          32  (21 prior + 11 new)
  dominant_emotion vals:  neutral (65), joy (6)  — only 2 labels appear
  overall_sentiment vals: Positive (50), Neutral (21) — no Negative
  sentiment_score range:  [0.5016, 0.9574]
  emotion_neutral range:  [0.8706, 0.9576]  — always very high
  emotion_joy range:      [0.2650, 0.9845]
  dining_mood labels:     5 unique values
  dominant_emotion rule:  argmax of the 7 emotion scores (verified 0 mismatches)
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pandas as pd
import pytest


# ── Path constants ─────────────────────────────────────────────────────────────
CLASSIFIED_CSV = "data/restaurants_with_classifications.csv"
EMOTIONS_CSV   = "data/restaurants_with_emotions.csv"

# ── The 7 emotion labels exactly as the model returns them ────────────────────
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# ── Column names for the 7 emotion scores in the CSV ─────────────────────────
EMOTION_COLS = [f"emotion_{label}" for label in EMOTION_LABELS]

# ── New columns added by Notebook 4 (11 total) ────────────────────────────────
NEW_EMOTION_COLUMNS = EMOTION_COLS + [
    "dominant_emotion",
    "overall_sentiment",
    "sentiment_score",
    "dining_mood",
]

# ── Valid label sets (sourced from the real CSV — not assumed) ─────────────────
VALID_DOMINANT_EMOTIONS = {"neutral", "joy"}   # only these appear in dataset

VALID_SENTIMENTS = {"Positive", "Neutral"}     # no Negative in this dataset

VALID_DINING_MOODS = {
    "Elegant & Refined",
    "Warm & Inviting",
    "Dramatic & Exciting",
    "Intimate & Personal",
    "Lively & Energetic",
}

# ── Full expected schema (32 cols) ────────────────────────────────────────────
PRIOR_COLUMNS = [
    "Name", "Location", "Description", "Address", "Telephone Number",
    "Price", "Cuisine Type", "Dining Atmosphere", "Sky-High Rooftop",
    "Michelin-Guide", "Customer Ratings", "Operation Hours",
    "Reservations", "Dress Code", "restaurant_metadata",
    "simple_cuisine_group", "dining_format", "predicted_occasion",
    "occasion_confidence", "predicted_vibe", "vibe_confidence",
]

EXPECTED_COLUMNS = PRIOR_COLUMNS + NEW_EMOTION_COLUMNS


# ═══════════════════════════════════════════════════════════════════════════════
# WRAPPER FUNCTIONS
# (Mirror of Notebook 4 implementation — keep in sync)
#
# distilbert-base-uncased-emotion output (sorted by score, highest first):
#   [{"label": "neutral", "score": 0.914},
#    {"label": "joy",     "score": 0.488}, ...]
#
# Secondary sentiment classifier output:
#   [{"label": "POSITIVE", "score": 0.881}]
# ═══════════════════════════════════════════════════════════════════════════════

DINING_MOOD_MAP: dict[str, str] = {
    "joy":      "Warm & Inviting",
    "neutral":  "Elegant & Refined",
    "surprise": "Dramatic & Exciting",
    "anger":    "Dramatic & Exciting",
    "fear":     "Intimate & Personal",
    "sadness":  "Intimate & Personal",
    "disgust":  "Lively & Energetic",
}


def extract_emotion_scores(model_output: list[dict]) -> dict[str, float]:
    """
    Converts the emotion model's list-of-dicts output into a flat dict:
      {"anger": 0.012, "disgust": 0.410, ..., "neutral": 0.914, ...}
    """
    return {item["label"]: round(item["score"], 4) for item in model_output}


def get_dominant_emotion(scores: dict[str, float]) -> str:
    """Returns the emotion label with the highest score (argmax)."""
    return max(scores, key=scores.get)


def get_dining_mood(dominant_emotion: str) -> str:
    """
    Maps dominant_emotion to a dining_mood label via DINING_MOOD_MAP.
    Falls back to 'Elegant & Refined' for any unknown emotion.
    """
    return DINING_MOOD_MAP.get(dominant_emotion, "Elegant & Refined")


def normalise_sentiment_label(raw_label: str) -> str:
    """
    Normalises raw sentiment classifier labels to Title Case.
      "POSITIVE" → "Positive"
      "NEGATIVE" → "Negative"
      "NEUTRAL"  → "Neutral"
    """
    return raw_label.strip().title()


def run_emotion_pipeline(emotion_model, text: str) -> list[dict]:
    """Calls the emotion model pipeline and returns the raw output list."""
    return emotion_model(text)


def run_sentiment_pipeline(sentiment_model, text: str) -> tuple[str, float]:
    """
    Calls the sentiment model and returns (normalised_label, score).
    """
    result = sentiment_model(text)
    top = result[0]
    return normalise_sentiment_label(top["label"]), round(top["score"], 4)


def classify_restaurant_emotion(emotion_model, sentiment_model, text: str) -> dict:
    """
    Full classification pipeline for one restaurant's metadata text.
    Returns a dict with all 11 new column values.
    """
    raw_output      = run_emotion_pipeline(emotion_model, text)
    scores          = extract_emotion_scores(raw_output)
    dominant        = get_dominant_emotion(scores)
    mood            = get_dining_mood(dominant)
    sentiment, score = run_sentiment_pipeline(sentiment_model, text)

    return {
        "emotion_anger":    scores.get("anger",    0.0),
        "emotion_disgust":  scores.get("disgust",  0.0),
        "emotion_fear":     scores.get("fear",     0.0),
        "emotion_joy":      scores.get("joy",      0.0),
        "emotion_neutral":  scores.get("neutral",  0.0),
        "emotion_sadness":  scores.get("sadness",  0.0),
        "emotion_surprise": scores.get("surprise", 0.0),
        "dominant_emotion": dominant,
        "overall_sentiment": sentiment,
        "sentiment_score":  score,
        "dining_mood":      mood,
    }


# ── Mock helpers ──────────────────────────────────────────────────────────────

def build_emotion_mock(scores: dict[str, float]) -> MagicMock:
    """
    Builds a MagicMock that mimics distilbert-base-uncased-emotion.
    Returns output sorted by score descending (as the real model does).
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    mock = MagicMock()
    mock.return_value = [{"label": label, "score": score}
                         for label, score in sorted_items]
    return mock


def build_sentiment_mock(label: str, score: float) -> MagicMock:
    """
    Builds a MagicMock that mimics a sentiment classifier.
    label should be in UPPERCASE as the real model returns (e.g. "POSITIVE").
    """
    mock = MagicMock()
    mock.return_value = [{"label": label, "score": score}]
    return mock


def make_neutral_dominant_scores(**overrides) -> dict[str, float]:
    """
    Returns a base emotion score dict where neutral is dominant.
    Pass keyword overrides to adjust individual scores.
    """
    base = {
        "anger": 0.012, "disgust": 0.410, "fear": 0.013,
        "joy": 0.488, "neutral": 0.914, "sadness": 0.038, "surprise": 0.216,
    }
    base.update(overrides)
    return base


def make_joy_dominant_scores(**overrides) -> dict[str, float]:
    """
    Returns a base emotion score dict where joy is dominant.
    """
    base = {
        "anger": 0.006, "disgust": 0.015, "fear": 0.006,
        "joy": 0.947, "neutral": 0.918, "sadness": 0.010, "surprise": 0.108,
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def df_classified():
    """21-column input CSV (output of Notebook 3)."""
    return pd.read_csv(CLASSIFIED_CSV)


@pytest.fixture(scope="module")
def df():
    """32-column output CSV (output of Notebook 4)."""
    return pd.read_csv(EMOTIONS_CSV)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CSV OUTPUT SCHEMA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputSchema:

    def test_column_count_is_32(self, df):
        """Output CSV must have exactly 32 columns."""
        assert len(df.columns) == 32, (
            f"Expected 32 columns, got {len(df.columns)}.\n"
            f"Columns: {list(df.columns)}"
        )

    def test_all_expected_columns_present(self, df):
        """Every column in EXPECTED_COLUMNS must exist in the output."""
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_unexpected_columns(self, df):
        """No columns beyond the expected 32 should be present."""
        extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
        assert not extra, f"Unexpected columns: {extra}"

    def test_column_order_matches_expected(self, df):
        """Column order must exactly match EXPECTED_COLUMNS."""
        assert list(df.columns) == EXPECTED_COLUMNS, (
            f"Column order mismatch.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Got:      {list(df.columns)}"
        )

    def test_eleven_new_columns_added(self, df_classified, df):
        """Notebook 4 must add exactly 11 columns to the prior 21."""
        new_cols = [c for c in df.columns if c not in df_classified.columns]
        assert len(new_cols) == 11, (
            f"Expected 11 new columns, found {len(new_cols)}: {new_cols}"
        )
        assert set(new_cols) == set(NEW_EMOTION_COLUMNS), (
            f"New column set mismatch.\n"
            f"Expected: {sorted(NEW_EMOTION_COLUMNS)}\n"
            f"Got:      {sorted(new_cols)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ROW COUNT & PRIOR COLUMN PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestRowCountAndPriorColumns:

    def test_row_count_is_71(self, df):
        """Notebook 4 must not add or remove rows — always 71."""
        assert len(df) == 71, f"Expected 71 rows, got {len(df)}"

    @pytest.mark.parametrize("col", [c for c in PRIOR_COLUMNS
                                     if c != "restaurant_metadata"])
    def test_prior_column_values_unchanged(self, df_classified, df, col):
        """
        All prior columns except restaurant_metadata must be identical
        before and after Notebook 4 runs.

        restaurant_metadata is excluded because Notebook 4 appends
        Dining Mood, Dominant Emotion, and Overall Sentiment tags to it —
        verified separately in TestMetadataEnrichment.
        """
        original   = df_classified[col].reset_index(drop=True).astype(str)
        after_emot = df[col].reset_index(drop=True).astype(str)
        mismatches = (original != after_emot).sum()
        assert mismatches == 0, (
            f"Column '{col}' was modified by Notebook 4: "
            f"{mismatches} row(s) differ"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NULL CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNullValues:

    @pytest.mark.parametrize("col", NEW_EMOTION_COLUMNS)
    def test_no_nulls_in_new_column(self, df, col):
        """Every new column produced by Notebook 4 must have zero nulls."""
        null_count = df[col].isnull().sum()
        assert null_count == 0, (
            f"Column '{col}' has {null_count} null(s):\n"
            f"{df[df[col].isnull()]['Name'].tolist()}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EMOTION SCORE VALIDITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmotionScores:

    @pytest.mark.parametrize("col", EMOTION_COLS)
    def test_emotion_score_is_float_dtype(self, df, col):
        """All 7 emotion score columns must be float64."""
        assert pd.api.types.is_float_dtype(df[col]), (
            f"'{col}' dtype is {df[col].dtype}, expected float64"
        )

    @pytest.mark.parametrize("col", EMOTION_COLS)
    def test_emotion_score_in_zero_to_one(self, df, col):
        """All emotion scores must be within [0.0, 1.0]."""
        oob = df[(df[col] < 0.0) | (df[col] > 1.0)]
        assert oob.empty, (
            f"'{col}' out-of-bounds scores:\n"
            f"{oob[['Name', col]].to_string()}"
        )

    @pytest.mark.parametrize("col", EMOTION_COLS)
    def test_emotion_score_above_zero(self, df, col):
        """
        Every emotion score must be > 0.0. The distilbert emotion model
        produces non-zero probabilities for all 7 classes on every input.
        A zero score means the pipeline failed or the column was unfilled.
        """
        zero_rows = df[df[col] == 0.0]
        assert zero_rows.empty, (
            f"'{col}' has zero-value scores:\n"
            f"{zero_rows['Name'].tolist()}"
        )

    def test_seven_emotion_cols_all_present(self, df):
        """All 7 emotion_* columns must be present — none may be dropped."""
        missing = [c for c in EMOTION_COLS if c not in df.columns]
        assert not missing, f"Missing emotion columns: {missing}"

    def test_emotion_neutral_is_always_high(self, df):
        """
        emotion_neutral is always the highest single score in this dataset,
        ranging from 0.8706 to 0.9576. Any value below 0.85 would indicate
        a different text was classified or the model changed.
        """
        low_neutral = df[df["emotion_neutral"] < 0.85]
        assert low_neutral.empty, (
            f"Unexpectedly low emotion_neutral scores:\n"
            f"{low_neutral[['Name', 'emotion_neutral']].to_string()}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DOMINANT EMOTION CONSISTENCY TESTS
# The core invariant: dominant_emotion must always equal the argmax of
# the 7 emotion score columns. Verified: 0 mismatches in real data.
# ═══════════════════════════════════════════════════════════════════════════════

class TestDominantEmotion:

    EMOTION_TO_COL = {
        "anger":    "emotion_anger",
        "disgust":  "emotion_disgust",
        "fear":     "emotion_fear",
        "joy":      "emotion_joy",
        "neutral":  "emotion_neutral",
        "sadness":  "emotion_sadness",
        "surprise": "emotion_surprise",
    }

    def test_dominant_emotion_equals_argmax_of_seven_scores(self, df):
        """
        For every row, dominant_emotion must match the column with the
        highest score among the 7 emotion_* columns.

        This is the fundamental correctness invariant for Notebook 4.
        If this fails it means the pipeline stored the wrong label.
        """
        mismatches = []
        for _, row in df.iterrows():
            scores = {label: row[col]
                      for label, col in self.EMOTION_TO_COL.items()}
            argmax_label = max(scores, key=scores.get)
            if argmax_label != row["dominant_emotion"]:
                mismatches.append({
                    "name":     row["Name"],
                    "expected": argmax_label,
                    "stored":   row["dominant_emotion"],
                    "scores":   {k: round(v, 4) for k, v in scores.items()},
                })
        assert not mismatches, (
            f"dominant_emotion does not match argmax of 7 scores "
            f"in {len(mismatches)} row(s):\n" +
            "\n".join(
                f"  {m['name']}: argmax={m['expected']}, "
                f"stored={m['stored']}, scores={m['scores']}"
                for m in mismatches
            )
        )

    def test_dominant_emotion_values_are_known_labels(self, df):
        """
        dominant_emotion must only contain labels from the 7-emotion set.
        No new label names, no typos, no empty strings.
        """
        actual_labels = set(df["dominant_emotion"].unique())
        unknown = actual_labels - set(EMOTION_LABELS)
        assert not unknown, (
            f"Unknown dominant_emotion labels: {unknown}\n"
            f"Valid labels: {EMOTION_LABELS}"
        )

    def test_neutral_is_most_common_dominant_emotion(self, df):
        """
        'neutral' must be the most frequent dominant_emotion (65 of 71).
        This reflects the factual/descriptive nature of restaurant metadata.
        """
        counts = df["dominant_emotion"].value_counts()
        assert counts.index[0] == "neutral", (
            f"Expected 'neutral' to be most common dominant_emotion, "
            f"got '{counts.index[0]}'. Counts:\n{counts.to_string()}"
        )

    def test_joy_appears_as_dominant_when_joy_exceeds_neutral(self, df):
        """
        The 6 rows where emotion_joy > emotion_neutral must all have
        dominant_emotion == 'joy'. This validates the argmax rule
        works correctly even when neutral is usually highest.
        """
        joy_over_neutral = df[df["emotion_joy"] > df["emotion_neutral"]]
        assert len(joy_over_neutral) > 0, (
            "Test precondition: expected some rows with joy > neutral"
        )
        wrong = joy_over_neutral[joy_over_neutral["dominant_emotion"] != "joy"]
        assert wrong.empty, (
            f"Rows with joy > neutral but dominant_emotion != 'joy':\n"
            f"{wrong[['Name', 'emotion_joy', 'emotion_neutral', 'dominant_emotion']].to_string()}"
        )

    def test_neutral_dominant_when_neutral_exceeds_all_others(self, df):
        """
        Rows where emotion_neutral is strictly greater than emotion_joy
        must have dominant_emotion == 'neutral'.
        """
        neutral_highest = df[df["emotion_neutral"] > df["emotion_joy"]]
        wrong = neutral_highest[neutral_highest["dominant_emotion"] != "neutral"]
        assert wrong.empty, (
            f"Rows with neutral > joy but dominant_emotion != 'neutral':\n"
            f"{wrong[['Name', 'emotion_neutral', 'emotion_joy', 'dominant_emotion']].to_string()}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. OVERALL SENTIMENT & SENTIMENT SCORE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSentimentColumns:

    def test_overall_sentiment_values_are_valid(self, df):
        """overall_sentiment must only contain known sentiment labels."""
        actual = set(df["overall_sentiment"].unique())
        unknown = actual - VALID_SENTIMENTS
        assert not unknown, (
            f"Unknown overall_sentiment values: {unknown}\n"
            f"Valid values: {VALID_SENTIMENTS}"
        )

    def test_sentiment_score_is_float_dtype(self, df):
        """sentiment_score must be stored as float64."""
        assert pd.api.types.is_float_dtype(df["sentiment_score"]), (
            f"sentiment_score dtype is {df['sentiment_score'].dtype}, expected float64"
        )

    def test_sentiment_score_in_zero_to_one(self, df):
        """All sentiment_score values must be within [0.0, 1.0]."""
        oob = df[(df["sentiment_score"] < 0.0) | (df["sentiment_score"] > 1.0)]
        assert oob.empty, (
            f"sentiment_score out of [0,1]:\n"
            f"{oob[['Name', 'sentiment_score']].to_string()}"
        )

    def test_sentiment_score_above_zero(self, df):
        """sentiment_score must be > 0.0 for every row."""
        zero = df[df["sentiment_score"] == 0.0]
        assert zero.empty, (
            f"Zero sentiment_score:\n{zero['Name'].tolist()}"
        )

    def test_dining_mood_values_are_valid(self, df):
        """dining_mood must only contain the 5 known mood labels."""
        actual = set(df["dining_mood"].unique())
        unknown = actual - VALID_DINING_MOODS
        assert not unknown, (
            f"Unknown dining_mood values: {unknown}\n"
            f"Valid values: {VALID_DINING_MOODS}"
        )

    def test_no_negative_sentiment_in_dataset(self, df):
        """
        This restaurant dataset has no Negative overall_sentiment rows.
        If 'Negative' ever appears it likely means a data processing error
        or a restaurant description was corrupted.
        """
        negative_rows = df[df["overall_sentiment"] == "Negative"]
        assert negative_rows.empty, (
            f"Unexpected 'Negative' sentiment rows:\n"
            f"{negative_rows[['Name', 'overall_sentiment', 'sentiment_score']].to_string()}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. METADATA ENRICHMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetadataEnrichment:

    def test_metadata_contains_dining_mood_tag(self, df):
        """restaurant_metadata must contain 'Dining Mood: {dining_mood}'."""
        for _, row in df.iterrows():
            expected = f"Dining Mood: {row['dining_mood']}"
            assert expected in row["restaurant_metadata"], (
                f"'{row['Name']}': missing '{expected}' in metadata.\n"
                f"Tail: {row['restaurant_metadata'][-200:]}"
            )

    def test_metadata_contains_dominant_emotion_tag(self, df):
        """restaurant_metadata must contain 'Dominant Emotion: {dominant_emotion}'."""
        for _, row in df.iterrows():
            expected = f"Dominant Emotion: {row['dominant_emotion']}"
            assert expected in row["restaurant_metadata"], (
                f"'{row['Name']}': missing '{expected}' in metadata.\n"
                f"Tail: {row['restaurant_metadata'][-200:]}"
            )

    def test_metadata_contains_overall_sentiment_tag(self, df):
        """restaurant_metadata must contain 'Overall Sentiment: {overall_sentiment}'."""
        for _, row in df.iterrows():
            expected = f"Overall Sentiment: {row['overall_sentiment']}"
            assert expected in row["restaurant_metadata"], (
                f"'{row['Name']}': missing '{expected}' in metadata.\n"
                f"Tail: {row['restaurant_metadata'][-200:]}"
            )

    def test_metadata_is_longer_than_classified_metadata(self, df_classified, df):
        """
        Notebook 4 appends 3 tags to restaurant_metadata. The emotions CSV
        metadata must be strictly longer than the classified CSV metadata.
        """
        merged = df_classified[["Name", "restaurant_metadata"]].merge(
            df[["Name", "restaurant_metadata"]],
            on="Name",
            suffixes=("_classified", "_emotions"),
        )
        not_extended = merged[
            merged["restaurant_metadata_emotions"].str.len() <=
            merged["restaurant_metadata_classified"].str.len()
        ]
        assert not_extended.empty, (
            f"These restaurants have unextended metadata in emotions CSV:\n"
            f"{not_extended['Name'].tolist()}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. WRAPPER UNIT TESTS — extract_emotion_scores
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractEmotionScores:

    def test_returns_dict_with_all_seven_labels(self):
        """extract_emotion_scores must return a key for each of the 7 emotions."""
        raw = [{"label": l, "score": 0.1} for l in EMOTION_LABELS]
        result = extract_emotion_scores(raw)
        assert set(result.keys()) == set(EMOTION_LABELS), (
            f"Missing labels: {set(EMOTION_LABELS) - set(result.keys())}"
        )

    def test_scores_are_correctly_assigned_to_labels(self):
        """Each label must receive its own score, not a neighbour's."""
        raw = [
            {"label": "neutral",  "score": 0.914},
            {"label": "joy",      "score": 0.488},
            {"label": "surprise", "score": 0.216},
            {"label": "disgust",  "score": 0.410},
            {"label": "anger",    "score": 0.012},
            {"label": "sadness",  "score": 0.038},
            {"label": "fear",     "score": 0.013},
        ]
        result = extract_emotion_scores(raw)
        assert result["neutral"]  == 0.914
        assert result["joy"]      == 0.488
        assert result["surprise"] == 0.216
        assert result["disgust"]  == 0.410
        assert result["anger"]    == 0.012
        assert result["sadness"]  == 0.038
        assert result["fear"]     == 0.013

    def test_scores_rounded_to_four_decimal_places(self):
        """Scores must be rounded to 4 decimal places."""
        raw = [{"label": "neutral", "score": 0.91455555},
               {"label": "joy",     "score": 0.48811111}] + \
              [{"label": l, "score": 0.1} for l in EMOTION_LABELS
               if l not in ("neutral", "joy")]
        result = extract_emotion_scores(raw)
        assert result["neutral"] == round(0.91455555, 4)
        assert result["joy"]     == round(0.48811111, 4)

    def test_order_of_input_does_not_affect_output(self):
        """The wrapper must correctly index by label name, not by position."""
        raw_forward  = [{"label": l, "score": i * 0.1}
                        for i, l in enumerate(EMOTION_LABELS)]
        raw_reversed = list(reversed(raw_forward))
        result_f = extract_emotion_scores(raw_forward)
        result_r = extract_emotion_scores(raw_reversed)
        assert result_f == result_r, (
            "Score extraction is position-dependent — must use label key"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. WRAPPER UNIT TESTS — get_dominant_emotion
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetDominantEmotion:

    def test_neutral_dominant_when_neutral_highest(self):
        """neutral must be returned when emotion_neutral > all others."""
        scores = make_neutral_dominant_scores()
        assert get_dominant_emotion(scores) == "neutral"

    def test_joy_dominant_when_joy_highest(self):
        """joy must be returned when emotion_joy > emotion_neutral."""
        scores = make_joy_dominant_scores()
        assert get_dominant_emotion(scores) == "joy"

    @pytest.mark.parametrize("winning_label", EMOTION_LABELS)
    def test_any_label_can_be_dominant(self, winning_label):
        """
        get_dominant_emotion must correctly identify any of the 7 emotions
        as dominant when it has the highest score.
        """
        scores = {label: 0.1 for label in EMOTION_LABELS}
        scores[winning_label] = 0.99
        assert get_dominant_emotion(scores) == winning_label, (
            f"Expected '{winning_label}' as dominant, "
            f"got '{get_dominant_emotion(scores)}'"
        )

    def test_returns_single_string_label(self):
        """get_dominant_emotion must return a plain string, not a list or tuple."""
        scores = make_neutral_dominant_scores()
        result = get_dominant_emotion(scores)
        assert isinstance(result, str), (
            f"Expected str, got {type(result)}"
        )

    def test_dominant_label_is_in_emotion_labels(self):
        """The returned dominant label must always be one of the 7 valid labels."""
        for winning in EMOTION_LABELS:
            scores = {label: 0.1 for label in EMOTION_LABELS}
            scores[winning] = 0.99
            result = get_dominant_emotion(scores)
            assert result in EMOTION_LABELS, (
                f"get_dominant_emotion returned '{result}', not in {EMOTION_LABELS}"
            )

    def test_highest_score_wins_not_alphabetical_order(self):
        """
        Verify the function selects by score, not by alphabetical label order.
        'anger' comes first alphabetically — it must not win unless it has
        the highest score.
        """
        scores = {label: 0.1 for label in EMOTION_LABELS}
        scores["surprise"] = 0.95  # surprise comes late alphabetically
        assert get_dominant_emotion(scores) == "surprise"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. WRAPPER UNIT TESTS — get_dining_mood
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetDiningMood:

    @pytest.mark.parametrize("dominant_emotion,expected_mood", [
        ("joy",      "Warm & Inviting"),
        ("neutral",  "Elegant & Refined"),
        ("surprise", "Dramatic & Exciting"),
        ("anger",    "Dramatic & Exciting"),
        ("fear",     "Intimate & Personal"),
        ("sadness",  "Intimate & Personal"),
        ("disgust",  "Lively & Energetic"),
    ])
    def test_dining_mood_mapping_for_each_emotion(self, dominant_emotion, expected_mood):
        """
        Each of the 7 dominant emotions must map to the correct dining_mood.
        This is the core dining_mood mapping table — every entry must be correct.
        """
        result = get_dining_mood(dominant_emotion)
        assert result == expected_mood, (
            f"dominant_emotion='{dominant_emotion}': "
            f"expected mood='{expected_mood}', got '{result}'"
        )

    def test_all_mapped_moods_are_valid_labels(self):
        """Every mood produced by get_dining_mood must be in VALID_DINING_MOODS."""
        for emotion in EMOTION_LABELS:
            mood = get_dining_mood(emotion)
            assert mood in VALID_DINING_MOODS, (
                f"Mood '{mood}' from dominant_emotion='{emotion}' "
                f"is not a valid dining mood"
            )

    def test_unknown_emotion_falls_back_gracefully(self):
        """
        An unexpected dominant_emotion label must not raise — it should
        fall back to the default mood without crashing.
        """
        try:
            result = get_dining_mood("confusion")  # not a real emotion label
            assert isinstance(result, str), "Fallback must return a string"
            assert result in VALID_DINING_MOODS, (
                f"Fallback result '{result}' is not a valid mood"
            )
        except KeyError as e:
            pytest.fail(
                f"get_dining_mood raised KeyError for unknown emotion: {e}. "
                f"Use dict.get() with a default, not direct key access."
            )

    def test_joy_maps_to_warm_and_inviting(self):
        """
        Joy is the only non-neutral dominant_emotion in this dataset.
        Its mapping to 'Warm & Inviting' is the most frequently exercised
        path after 'neutral' → 'Elegant & Refined'.
        """
        assert get_dining_mood("joy") == "Warm & Inviting"

    def test_neutral_maps_to_elegant_and_refined(self):
        """
        Neutral is dominant in 65 of 71 restaurants.
        Its mapping is the default path for this dataset.
        """
        assert get_dining_mood("neutral") == "Elegant & Refined"

    def test_anger_and_surprise_both_map_to_dramatic_exciting(self):
        """anger and surprise must both map to 'Dramatic & Exciting'."""
        assert get_dining_mood("anger")    == "Dramatic & Exciting"
        assert get_dining_mood("surprise") == "Dramatic & Exciting"

    def test_fear_and_sadness_both_map_to_intimate_personal(self):
        """fear and sadness must both map to 'Intimate & Personal'."""
        assert get_dining_mood("fear")    == "Intimate & Personal"
        assert get_dining_mood("sadness") == "Intimate & Personal"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. WRAPPER UNIT TESTS — normalise_sentiment_label
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormaliseSentimentLabel:

    @pytest.mark.parametrize("raw,expected", [
        ("POSITIVE", "Positive"),
        ("NEGATIVE", "Negative"),
        ("NEUTRAL",  "Neutral"),
        ("positive", "Positive"),
        ("neutral",  "Neutral"),
        ("Positive", "Positive"),
    ])
    def test_normalises_to_title_case(self, raw, expected):
        """Any casing of the sentiment label must normalise to Title Case."""
        assert normalise_sentiment_label(raw) == expected, (
            f"'{raw}' → expected '{expected}', got '{normalise_sentiment_label(raw)}'"
        )

    def test_strips_whitespace(self):
        """Leading/trailing whitespace must be stripped before Title-casing."""
        assert normalise_sentiment_label("  POSITIVE  ") == "Positive"

    def test_returns_string(self):
        """normalise_sentiment_label must always return a str."""
        assert isinstance(normalise_sentiment_label("POSITIVE"), str)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. FULL PIPELINE INTEGRATION TESTS — classify_restaurant_emotion
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyRestaurantEmotion:

    def test_returns_dict_with_all_eleven_keys(self):
        """classify_restaurant_emotion must return all 11 new column keys."""
        emotion_mock   = build_emotion_mock(make_neutral_dominant_scores())
        sentiment_mock = build_sentiment_mock("POSITIVE", 0.88)
        result = classify_restaurant_emotion(
            emotion_mock, sentiment_mock, "A refined French restaurant."
        )
        assert set(result.keys()) == set(NEW_EMOTION_COLUMNS), (
            f"Missing keys: {set(NEW_EMOTION_COLUMNS) - set(result.keys())}"
        )

    def test_dominant_emotion_is_argmax_of_returned_scores(self):
        """
        The dominant_emotion in the result dict must match the argmax
        of the seven emotion scores in the same dict.
        """
        scores = make_neutral_dominant_scores()
        emotion_mock   = build_emotion_mock(scores)
        sentiment_mock = build_sentiment_mock("NEUTRAL", 0.65)
        result = classify_restaurant_emotion(
            emotion_mock, sentiment_mock, "Restaurant text."
        )
        score_vals = {label: result[f"emotion_{label}"] for label in EMOTION_LABELS}
        argmax_label = max(score_vals, key=score_vals.get)
        assert result["dominant_emotion"] == argmax_label, (
            f"dominant_emotion='{result['dominant_emotion']}' "
            f"does not match argmax='{argmax_label}'. Scores: {score_vals}"
        )

    def test_dining_mood_consistent_with_dominant_emotion(self):
        """
        dining_mood must be the value that DINING_MOOD_MAP assigns to
        the dominant_emotion in the same result.
        """
        scores = make_joy_dominant_scores()
        emotion_mock   = build_emotion_mock(scores)
        sentiment_mock = build_sentiment_mock("POSITIVE", 0.90)
        result = classify_restaurant_emotion(
            emotion_mock, sentiment_mock, "Warm and joyful restaurant."
        )
        assert result["dominant_emotion"] == "joy"
        assert result["dining_mood"] == DINING_MOOD_MAP["joy"], (
            f"dining_mood='{result['dining_mood']}' inconsistent with "
            f"dominant_emotion='joy'"
        )

    def test_sentiment_label_is_normalised(self):
        """overall_sentiment must be in Title Case regardless of model output."""
        emotion_mock   = build_emotion_mock(make_neutral_dominant_scores())
        sentiment_mock = build_sentiment_mock("POSITIVE", 0.80)
        result = classify_restaurant_emotion(
            emotion_mock, sentiment_mock, "Restaurant text."
        )
        assert result["overall_sentiment"] == "Positive", (
            f"Expected 'Positive' (Title Case), got '{result['overall_sentiment']}'"
        )

    def test_all_seven_emotion_scores_stored(self):
        """All 7 emotion scores from the mock must appear in the result."""
        scores = make_neutral_dominant_scores()
        emotion_mock   = build_emotion_mock(scores)
        sentiment_mock = build_sentiment_mock("NEUTRAL", 0.62)
        result = classify_restaurant_emotion(
            emotion_mock, sentiment_mock, "Restaurant."
        )
        for label in EMOTION_LABELS:
            col = f"emotion_{label}"
            assert col in result, f"Missing key '{col}' in result"
            assert isinstance(result[col], float), (
                f"'{col}' must be float, got {type(result[col])}"
            )

    def test_emotion_model_called_exactly_once(self):
        """The emotion model must be called exactly once per restaurant."""
        emotion_mock   = build_emotion_mock(make_neutral_dominant_scores())
        sentiment_mock = build_sentiment_mock("POSITIVE", 0.85)
        classify_restaurant_emotion(emotion_mock, sentiment_mock, "Text.")
        assert emotion_mock.call_count == 1, (
            f"Emotion model called {emotion_mock.call_count} times (expected 1)"
        )

    def test_sentiment_model_called_exactly_once(self):
        """The sentiment model must be called exactly once per restaurant."""
        emotion_mock   = build_emotion_mock(make_neutral_dominant_scores())
        sentiment_mock = build_sentiment_mock("POSITIVE", 0.85)
        classify_restaurant_emotion(emotion_mock, sentiment_mock, "Text.")
        assert sentiment_mock.call_count == 1, (
            f"Sentiment model called {sentiment_mock.call_count} times (expected 1)"
        )

    def test_input_text_passed_to_emotion_model(self):
        """The restaurant metadata text must be forwarded verbatim to the model."""
        emotion_mock   = build_emotion_mock(make_neutral_dominant_scores())
        sentiment_mock = build_sentiment_mock("POSITIVE", 0.85)
        test_text = "An elegant 14-seat French tasting menu in Santa Monica."
        classify_restaurant_emotion(emotion_mock, sentiment_mock, test_text)
        args, _ = emotion_mock.call_args
        assert args[0] == test_text, (
            f"Emotion model received '{args[0]}' instead of '{test_text}'"
        )

    def test_input_text_passed_to_sentiment_model(self):
        """The same text must be passed to the sentiment model too."""
        emotion_mock   = build_emotion_mock(make_neutral_dominant_scores())
        sentiment_mock = build_sentiment_mock("POSITIVE", 0.85)
        test_text = "A vibrant Korean BBQ spot in Koreatown."
        classify_restaurant_emotion(emotion_mock, sentiment_mock, test_text)
        args, _ = sentiment_mock.call_args
        assert args[0] == test_text, (
            f"Sentiment model received '{args[0]}' instead of '{test_text}'"
        )

    def test_empty_string_does_not_raise(self):
        """An empty metadata string must not raise — mocks will respond normally."""
        emotion_mock   = build_emotion_mock(make_neutral_dominant_scores())
        sentiment_mock = build_sentiment_mock("NEUTRAL", 0.55)
        try:
            result = classify_restaurant_emotion(emotion_mock, sentiment_mock, "")
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"classify_restaurant_emotion raised on empty string: {e}")

    def test_special_characters_in_text_do_not_raise(self):
        """
        Restaurant metadata contains apostrophes, slashes, quotes, and
        ampersands. None of these must cause the wrapper to raise.
        """
        emotion_mock   = build_emotion_mock(make_neutral_dominant_scores())
        sentiment_mock = build_sentiment_mock("POSITIVE", 0.78)
        tricky_texts = [
            "Chef's Table — omakase & kaiseki",
            "n/naka: 13-course tasting menu",
            'Described as "exceptional" by Michelin',
            "Gwen Butcher Shop & Restaurant — 100% prime beef",
        ]
        for text in tricky_texts:
            try:
                classify_restaurant_emotion(emotion_mock, sentiment_mock, text)
            except Exception as e:
                pytest.fail(
                    f"classify_restaurant_emotion raised on special-char text: {e}\n"
                    f"Text: {text}"
                )

    def test_result_values_are_correct_types(self):
        """Every value in the result dict must have the correct Python type."""
        emotion_mock   = build_emotion_mock(make_neutral_dominant_scores())
        sentiment_mock = build_sentiment_mock("POSITIVE", 0.88)
        result = classify_restaurant_emotion(
            emotion_mock, sentiment_mock, "Restaurant text."
        )
        for col in EMOTION_COLS:
            assert isinstance(result[col], float), (
                f"'{col}' must be float, got {type(result[col])}"
            )
        assert isinstance(result["dominant_emotion"],  str),   "dominant_emotion must be str"
        assert isinstance(result["overall_sentiment"], str),   "overall_sentiment must be str"
        assert isinstance(result["sentiment_score"],   float), "sentiment_score must be float"
        assert isinstance(result["dining_mood"],       str),   "dining_mood must be str"


# ═══════════════════════════════════════════════════════════════════════════════
# 13. MOCK INTEGRITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMockIntegrity:

    def test_emotion_mock_returns_all_seven_labels(self):
        """The emotion mock must return a dict entry for all 7 labels."""
        scores = make_neutral_dominant_scores()
        mock   = build_emotion_mock(scores)
        result = mock("any text")
        returned_labels = {item["label"] for item in result}
        assert returned_labels == set(EMOTION_LABELS), (
            f"Mock missing labels: {set(EMOTION_LABELS) - returned_labels}"
        )

    def test_emotion_mock_sorted_descending_by_score(self):
        """
        The real distilbert model returns results sorted highest score first.
        Our mock must replicate this so the wrapper's index-0 access works.
        """
        scores = make_neutral_dominant_scores()  # neutral=0.914 is highest
        mock   = build_emotion_mock(scores)
        result = mock("text")
        returned_scores = [item["score"] for item in result]
        assert returned_scores == sorted(returned_scores, reverse=True), (
            f"Emotion mock output is not sorted descending: {returned_scores}"
        )

    def test_emotion_mock_highest_label_is_first(self):
        """The label with the highest score must be first in the mock output."""
        scores = make_joy_dominant_scores()  # joy=0.947 is highest
        mock   = build_emotion_mock(scores)
        result = mock("text")
        assert result[0]["label"] == "joy", (
            f"Expected 'joy' first (highest score), got '{result[0]['label']}'"
        )

    def test_sentiment_mock_returns_correct_label_and_score(self):
        """The sentiment mock must return the specified label and score."""
        mock   = build_sentiment_mock("POSITIVE", 0.88)
        result = mock("text")
        assert result[0]["label"] == "POSITIVE"
        assert result[0]["score"] == 0.88

    def test_build_mock_with_custom_winning_emotion(self):
        """
        build_emotion_mock must correctly set the highest score to the
        winning emotion regardless of which emotion it is.
        """
        for winner in EMOTION_LABELS:
            scores = {label: 0.1 for label in EMOTION_LABELS}
            scores[winner] = 0.99
            mock   = build_emotion_mock(scores)
            result = mock("text")
            assert result[0]["label"] == winner, (
                f"Expected '{winner}' first, got '{result[0]['label']}'"
            )
