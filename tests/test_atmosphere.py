"""
tests/test_atmosphere.py
=========================
Unit tests for Notebook 4 — Restaurant Atmosphere Classification.

Strategy
--------
The zero-shot classification model is fully mocked. No downloads, no GPU,
no network.

  - MoritzLaurer/deberta-v3-base-zeroshot-v2.0 (zero-shot-classification)
      Called with multi_label=True against 13 custom atmosphere labels.
      Returns a dict: {"sequence": str, "labels": [...], "scores": [...]}
      sorted by score descending. The wrapper picks labels[0] /
      scores[0] as predicted_atmosphere / atmosphere_confidence, and
      labels[1] as secondary_atmosphere.

This replaces the old emotion/sentiment model entirely — there is no
distilbert-base-uncased-emotion or secondary sentiment classifier in
this pipeline. Zero-shot NLI lets the model score arbitrary custom
labels (Romantic, Fine Dining/Formal, Cozy/Intimate, etc.) without any
fine-tuning, which a fixed-label emotion classifier cannot do.

Tests operate in two layers:

  Layer 1 — CSV artifact tests (Sections 1–6):
      Load restaurants_with_atmosphere.csv and assert schema, row count,
      nulls, label validity, confidence score bounds, and metadata
      enrichment tags.

  Layer 2 — Wrapper unit tests (Sections 7–12):
      Test the wrapper functions directly with synthetic mock outputs,
      covering primary/secondary atmosphere selection, confidence
      extraction, multi-label scoring, and edge cases.

Key facts from the real CSV (grounded, not assumed):
  Total rows:               94
  Total columns:            24  (21 prior + 3 new)
  simple_cuisine_group:     19 nulls (restaurants added after the cuisine
                             classification pass — see TestNullValues note)
  predicted_atmosphere vals: 7 of 13 possible labels appear as primary —
                             Fine Dining / Formal (53), Casual (15),
                             Trendy / Hip (14), Romantic (6), Fine Casual (2),
                             Minimalist / Modern (2), Upscale Casual (2)
  secondary_atmosphere vals: 12 of 13 possible labels appear as secondary
  atmosphere_confidence range: [0.5024, 0.9994]
  primary == secondary:     0 rows (verified — the two are always distinct)
  multi-location names:     Pine & Crane (2 rows), Badmaash (2 rows)

NOTE ON ROW COUNT vs. UPSTREAM CSV:
  The dataset grew from 71 to 94 restaurants during this phase, so
  restaurants_with_classifications.csv (71 rows, the old upstream input)
  and restaurants_with_atmosphere.csv (94 rows, current output) are NOT
  row-aligned. Unlike test_sentiment.py, this file does NOT test "prior
  column values unchanged" against the old upstream CSV — that
  comparison is not meaningful when the row count itself changed mid-
  phase. All tests here validate restaurants_with_atmosphere.csv as a
  standalone artifact.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pandas as pd
import pytest


# ── Path constants ─────────────────────────────────────────────────────────────
ATMOSPHERE_CSV = "data/restaurants_with_atmosphere.csv"

# ── The 13 atmosphere labels exactly as defined in Notebook 4 ──────────────────
# Sourced from Restaurant_Atmosphere_List.docx. Family-Friendly and
# Rustic/Farmhouse are excluded — no restaurant in the dataset matches
# either profile.
ATMOSPHERE_LABELS = [
    "Romantic",
    "Energetic / Lively",
    "Casual",
    "Fine Casual",
    "Fine Dining / Formal",
    "Cozy / Intimate",
    "Trendy / Hip",
    "Industrial / Urban",
    "Minimalist / Modern",
    "Traditional / Classic",
    "Theatrical / Entertainment",
    "Beachy / Tropical",
    "Upscale Casual",
]

# ── New columns added by Notebook 4 (3 total) ──────────────────────────────────
NEW_ATMOSPHERE_COLUMNS = [
    "predicted_atmosphere",
    "atmosphere_confidence",
    "secondary_atmosphere",
]

# ── Labels actually observed as predicted_atmosphere in the real CSV ──────────
# (7 of the 13 possible labels appear as a top prediction in this dataset;
# the rest may still appear as secondary_atmosphere or in future runs as
# the dataset grows.)
OBSERVED_PRIMARY_LABELS = {
    "Fine Dining / Formal",
    "Casual",
    "Trendy / Hip",
    "Romantic",
    "Fine Casual",
    "Minimalist / Modern",
    "Upscale Casual",
}

# ── Full expected schema (24 cols: 21 prior + 3 new) ───────────────────────────
PRIOR_COLUMNS = [
    "Name", "Location", "Description", "Address", "Telephone Number",
    "Price", "Cuisine Type", "Dining Atmosphere", "Sky-High Rooftop",
    "Michelin-Guide", "Customer Ratings", "Operation Hours",
    "Reservations", "Dress Code", "restaurant_metadata",
    "simple_cuisine_group", "dining_format", "predicted_occasion",
    "occasion_confidence", "predicted_vibe", "vibe_confidence",
]

EXPECTED_COLUMNS = PRIOR_COLUMNS + NEW_ATMOSPHERE_COLUMNS


# ═══════════════════════════════════════════════════════════════════════════════
# WRAPPER FUNCTIONS
# (Mirror of Notebook 4 implementation — keep in sync)
#
# MoritzLaurer/deberta-v3-base-zeroshot-v2.0 output (multi_label=True,
# sorted by score, highest first):
#   {
#     "sequence": "<restaurant_metadata text>",
#     "labels":  ["Fine Dining / Formal", "Upscale Casual", ...],
#     "scores":  [0.9989, 0.7421, ...],
#   }
# ═══════════════════════════════════════════════════════════════════════════════

def run_zero_shot_atmosphere(classifier, text: str, candidate_labels: list[str]) -> dict:
    """Calls the zero-shot atmosphere classifier and returns the raw output dict."""
    return classifier(text, candidate_labels=candidate_labels, multi_label=True)


def classify_atmosphere(classifier, text: str) -> tuple[str, float, str]:
    """
    Classifies a restaurant's atmosphere. Returns
    (predicted_atmosphere, atmosphere_confidence, secondary_atmosphere).

    predicted_atmosphere  : labels[0]  — top-scoring label
    atmosphere_confidence : scores[0], rounded to 4 decimal places
    secondary_atmosphere  : labels[1]  — runner-up label
    """
    result = run_zero_shot_atmosphere(classifier, text, ATMOSPHERE_LABELS)
    predicted_atmosphere  = result["labels"][0]
    atmosphere_confidence = round(result["scores"][0], 4)
    secondary_atmosphere  = result["labels"][1]
    return predicted_atmosphere, atmosphere_confidence, secondary_atmosphere


def append_atmosphere_tags(metadata_text: str, predicted: str, confidence: float, secondary: str) -> str:
    """
    Appends the three atmosphere tags to restaurant_metadata, mirroring
    Notebook 4's metadata enrichment step.
    """
    tags = (
        f" Primary Atmosphere: {predicted}."
        f" Secondary Atmosphere: {secondary}."
        f" Atmosphere Confidence: {confidence}."
    )
    return metadata_text + tags


# ── Mock helpers ──────────────────────────────────────────────────────────────

def build_atmosphere_mock(scores: dict[str, float]) -> MagicMock:
    """
    Builds a MagicMock that mimics MoritzLaurer/deberta-v3-base-zeroshot-v2.0
    called with multi_label=True. Returns output sorted by score descending,
    as the real pipeline does.
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    mock = MagicMock()
    mock.return_value = {
        "sequence": "mock restaurant metadata text",
        "labels":  [label for label, _ in sorted_items],
        "scores":  [score for _, score in sorted_items],
    }
    return mock


def make_fine_dining_dominant_scores(**overrides) -> dict[str, float]:
    """Returns a base atmosphere score dict where Fine Dining / Formal is dominant."""
    base = {label: 0.05 for label in ATMOSPHERE_LABELS}
    base["Fine Dining / Formal"] = 0.9989
    base["Upscale Casual"]       = 0.7421
    base["Traditional / Classic"] = 0.3012
    base.update(overrides)
    return base


def make_romantic_dominant_scores(**overrides) -> dict[str, float]:
    """Returns a base atmosphere score dict where Romantic is dominant."""
    base = {label: 0.04 for label in ATMOSPHERE_LABELS}
    base["Romantic"]        = 0.9512
    base["Cozy / Intimate"] = 0.6834
    base["Fine Dining / Formal"] = 0.2210
    base.update(overrides)
    return base


def make_trendy_dominant_scores(**overrides) -> dict[str, float]:
    """Returns a base atmosphere score dict where Trendy / Hip is dominant."""
    base = {label: 0.03 for label in ATMOSPHERE_LABELS}
    base["Trendy / Hip"]        = 0.8843
    base["Industrial / Urban"]  = 0.5102
    base["Energetic / Lively"]  = 0.4017
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def df():
    """24-column output CSV (output of Notebook 4)."""
    return pd.read_csv(ATMOSPHERE_CSV)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CSV OUTPUT SCHEMA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputSchema:

    def test_column_count_is_24(self, df):
        """Output CSV must have exactly 24 columns."""
        assert len(df.columns) == 24, (
            f"Expected 24 columns, got {len(df.columns)}.\n"
            f"Columns: {list(df.columns)}"
        )

    def test_all_expected_columns_present(self, df):
        """Every column in EXPECTED_COLUMNS must exist in the output."""
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_unexpected_columns(self, df):
        """No columns beyond the expected 24 should be present."""
        extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
        assert not extra, f"Unexpected columns: {extra}"

    def test_column_order_matches_expected(self, df):
        """Column order must exactly match EXPECTED_COLUMNS."""
        assert list(df.columns) == EXPECTED_COLUMNS, (
            f"Column order mismatch.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Got:      {list(df.columns)}"
        )

    def test_three_new_atmosphere_columns_present(self, df):
        """Notebook 4 must add exactly the 3 atmosphere columns."""
        new_cols = [c for c in df.columns if c not in PRIOR_COLUMNS]
        assert len(new_cols) == 3, (
            f"Expected 3 new columns, found {len(new_cols)}: {new_cols}"
        )
        assert set(new_cols) == set(NEW_ATMOSPHERE_COLUMNS), (
            f"New column set mismatch.\n"
            f"Expected: {sorted(NEW_ATMOSPHERE_COLUMNS)}\n"
            f"Got:      {sorted(new_cols)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ROW COUNT TESTS
#
# NOTE: Unlike test_sentiment.py, this file does NOT compare row-by-row
# against an upstream CSV. The dataset grew from 71 to 94 restaurants
# during this phase, so restaurants_with_classifications.csv (71 rows)
# and restaurants_with_atmosphere.csv (94 rows) are not row-aligned —
# any "prior column unchanged" comparison would be comparing the wrong
# rows to each other. All checks here validate the 94-row CSV standalone.
# ═══════════════════════════════════════════════════════════════════════════════

class TestRowCount:

    def test_row_count_is_94(self, df):
        """Dataset must contain exactly 94 restaurant records."""
        assert len(df) == 94, f"Expected 94 rows, got {len(df)}"

    def test_multi_location_pine_and_crane_has_two_rows(self, df):
        """Pine & Crane intentionally has two locations — both rows must be present."""
        count = len(df[df["Name"] == "Pine & Crane"])
        assert count == 2, f"Expected 2 rows for Pine & Crane, found {count}"

    def test_multi_location_badmaash_has_two_rows(self, df):
        """Badmaash intentionally has two locations — both rows must be present."""
        count = len(df[df["Name"] == "Badmaash"])
        assert count == 2, f"Expected 2 rows for Badmaash, found {count}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NULL CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNullValues:

    @pytest.mark.parametrize("col", NEW_ATMOSPHERE_COLUMNS)
    def test_no_nulls_in_new_atmosphere_column(self, df, col):
        """
        Every new atmosphere column produced by Notebook 4 must have zero
        nulls — including for the 23 restaurants added after the original
        71-row cuisine classification pass. The zero-shot atmosphere model
        runs directly on restaurant_metadata text and does not depend on
        simple_cuisine_group, so it has no upstream-null dependency.
        """
        null_count = df[col].isnull().sum()
        assert null_count == 0, (
            f"Column '{col}' has {null_count} null(s):\n"
            f"{df[df[col].isnull()]['Name'].tolist()}"
        )

    def test_simple_cuisine_group_nulls_are_a_known_upstream_gap(self, df):
        """
        simple_cuisine_group has 19 known nulls — restaurants added to the
        dataset after Notebook 3's classification pass. This is NOT a
        Notebook 4 defect (atmosphere classification doesn't touch this
        column), but it's documented here so a future schema change that
        fixes the upstream gap doesn't go unnoticed.
        """
        null_count = df["simple_cuisine_group"].isnull().sum()
        assert null_count == 19, (
            f"Expected exactly 19 known nulls in simple_cuisine_group "
            f"(restaurants added after Notebook 3), got {null_count}. "
            f"If this is now 0, the upstream gap has been fixed — "
            f"consider updating this test's expectation."
        )

    def test_no_empty_string_in_predicted_atmosphere(self, df):
        """predicted_atmosphere must not contain empty strings."""
        empty = (df["predicted_atmosphere"].astype(str).str.strip() == "").sum()
        assert empty == 0, f"predicted_atmosphere has {empty} empty string(s)"

    def test_no_empty_string_in_secondary_atmosphere(self, df):
        """secondary_atmosphere must not contain empty strings."""
        empty = (df["secondary_atmosphere"].astype(str).str.strip() == "").sum()
        assert empty == 0, f"secondary_atmosphere has {empty} empty string(s)"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ATMOSPHERE LABEL VALIDITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAtmosphereLabelValidity:

    def test_all_predicted_atmosphere_values_are_known_labels(self, df):
        """
        Every value in predicted_atmosphere must be one of the 13 known
        atmosphere labels. Unknown values would mean a label was renamed
        or a typo was introduced in Notebook 4 without updating this test.
        """
        actual = set(df["predicted_atmosphere"].unique())
        unknown = actual - set(ATMOSPHERE_LABELS)
        assert not unknown, (
            f"Unknown predicted_atmosphere labels: {unknown}\n"
            f"Valid labels: {ATMOSPHERE_LABELS}"
        )

    def test_all_secondary_atmosphere_values_are_known_labels(self, df):
        """Every value in secondary_atmosphere must also be a known label."""
        actual = set(df["secondary_atmosphere"].unique())
        unknown = actual - set(ATMOSPHERE_LABELS)
        assert not unknown, (
            f"Unknown secondary_atmosphere labels: {unknown}\n"
            f"Valid labels: {ATMOSPHERE_LABELS}"
        )

    def test_observed_primary_labels_match_real_dataset(self, df):
        """
        Exactly the 7 labels recorded in OBSERVED_PRIMARY_LABELS should
        appear as predicted_atmosphere in this dataset. If new restaurants
        push a previously-unseen label into the primary slot, that's not
        an error — but it's worth knowing about, so this test documents
        the current state rather than silently allowing drift.
        """
        actual = set(df["predicted_atmosphere"].unique())
        assert actual == OBSERVED_PRIMARY_LABELS, (
            f"predicted_atmosphere label set has changed.\n"
            f"Previously observed: {sorted(OBSERVED_PRIMARY_LABELS)}\n"
            f"Now observed:        {sorted(actual)}\n"
            f"This may be expected if restaurants were added/removed — "
            f"update OBSERVED_PRIMARY_LABELS if so."
        )

    def test_fine_dining_formal_is_most_common_primary_atmosphere(self, df):
        """
        'Fine Dining / Formal' must be the most frequent predicted_atmosphere
        (53 of 94 in the real dataset). This reflects the upscale/Michelin-
        leaning skew of the restaurant guide this project is built from.
        """
        counts = df["predicted_atmosphere"].value_counts()
        assert counts.index[0] == "Fine Dining / Formal", (
            f"Expected 'Fine Dining / Formal' to be most common, "
            f"got '{counts.index[0]}'. Counts:\n{counts.to_string()}"
        )

    def test_primary_and_secondary_atmosphere_never_identical(self, df):
        """
        predicted_atmosphere and secondary_atmosphere must always differ —
        the wrapper takes labels[0] and labels[1] from the zero-shot
        output, which are by definition the top-2 distinct scoring labels.
        Verified: 0 matching rows in the real 94-row dataset.
        """
        same = df[df["predicted_atmosphere"] == df["secondary_atmosphere"]]
        assert same.empty, (
            f"{len(same)} row(s) have identical primary and secondary "
            f"atmosphere — the wrapper may be returning duplicate labels:\n"
            f"{same[['Name', 'predicted_atmosphere', 'secondary_atmosphere']].to_string()}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONFIDENCE SCORE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceScores:

    def test_atmosphere_confidence_is_float_dtype(self, df):
        """atmosphere_confidence must be stored as float64, not object/string."""
        assert pd.api.types.is_float_dtype(df["atmosphere_confidence"]), (
            f"atmosphere_confidence dtype is {df['atmosphere_confidence'].dtype}, "
            f"expected float64"
        )

    def test_atmosphere_confidence_in_zero_to_one(self, df):
        """All atmosphere_confidence scores must be within [0.0, 1.0]."""
        oob = df[
            (df["atmosphere_confidence"] < 0.0) |
            (df["atmosphere_confidence"] > 1.0)
        ]
        assert oob.empty, (
            f"atmosphere_confidence out of [0,1] bounds:\n"
            f"{oob[['Name', 'atmosphere_confidence']].to_string()}"
        )

    def test_atmosphere_confidence_above_floor(self, df):
        """
        All atmosphere_confidence scores must be > 0.0 — a zero confidence
        means the pipeline returned nonsense or defaulted without classifying.
        """
        zero_conf = df[df["atmosphere_confidence"] == 0.0]
        assert zero_conf.empty, (
            f"Zero atmosphere_confidence found:\n"
            f"{zero_conf[['Name', 'atmosphere_confidence']].to_string()}"
        )

    def test_atmosphere_confidence_range_matches_real_data(self, df):
        """
        atmosphere_confidence in this dataset ranges from 0.5024 to 0.9994.
        A value far outside this range (e.g. negative, or > 1.0) would
        indicate the model or rounding logic changed.
        """
        min_conf = df["atmosphere_confidence"].min()
        max_conf = df["atmosphere_confidence"].max()
        assert min_conf >= 0.50, (
            f"Unexpectedly low minimum atmosphere_confidence: {min_conf}"
        )
        assert max_conf <= 1.00, (
            f"Unexpectedly high maximum atmosphere_confidence: {max_conf}"
        )

    def test_high_confidence_rows_are_the_majority(self, df):
        """
        The 25th percentile of atmosphere_confidence is ~0.95 in this
        dataset, meaning at least 75% of restaurants are classified with
        very high confidence. This sanity-checks that the zero-shot model
        is producing decisive (not wishy-washy) predictions on this data.
        """
        high_conf_count = (df["atmosphere_confidence"] >= 0.90).sum()
        proportion = high_conf_count / len(df)
        assert proportion >= 0.70, (
            f"Only {proportion:.1%} of restaurants have confidence >= 0.90, "
            f"expected at least 70%. Model behavior may have changed."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. METADATA ENRICHMENT TESTS
# Notebook 4 appends 3 atmosphere tags to restaurant_metadata.
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetadataEnrichment:

    def test_metadata_contains_primary_atmosphere_tag(self, df):
        """
        restaurant_metadata must contain 'Primary Atmosphere: {predicted_atmosphere}'
        for every row.
        """
        for _, row in df.iterrows():
            expected = f"Primary Atmosphere: {row['predicted_atmosphere']}"
            assert expected in row["restaurant_metadata"], (
                f"'{row['Name']}': expected '{expected}' in metadata.\n"
                f"Metadata tail: {row['restaurant_metadata'][-200:]}"
            )

    def test_metadata_contains_secondary_atmosphere_tag(self, df):
        """
        restaurant_metadata must contain 'Secondary Atmosphere: {secondary_atmosphere}'
        for every row.
        """
        for _, row in df.iterrows():
            expected = f"Secondary Atmosphere: {row['secondary_atmosphere']}"
            assert expected in row["restaurant_metadata"], (
                f"'{row['Name']}': expected '{expected}' in metadata.\n"
                f"Metadata tail: {row['restaurant_metadata'][-200:]}"
            )

    def test_metadata_contains_atmosphere_confidence_tag(self, df):
        """
        restaurant_metadata must contain 'Atmosphere Confidence: {atmosphere_confidence}'
        for every row.
        """
        for _, row in df.iterrows():
            expected = f"Atmosphere Confidence: {row['atmosphere_confidence']}"
            assert expected in row["restaurant_metadata"], (
                f"'{row['Name']}': expected '{expected}' in metadata.\n"
                f"Metadata tail: {row['restaurant_metadata'][-200:]}"
            )

    def test_metadata_atmosphere_tags_appear_in_correct_order(self, df):
        """
        The three atmosphere tags must appear in the order Primary →
        Secondary → Confidence, matching Notebook 4's append_atmosphere_tags
        implementation. Order matters for readability and for any future
        regex-based parsing of the metadata string.
        """
        sample = df.iloc[0]
        metadata = sample["restaurant_metadata"]
        primary_idx    = metadata.find("Primary Atmosphere:")
        secondary_idx  = metadata.find("Secondary Atmosphere:")
        confidence_idx = metadata.find("Atmosphere Confidence:")
        assert primary_idx < secondary_idx < confidence_idx, (
            f"Atmosphere tags out of order in metadata.\n"
            f"Primary at {primary_idx}, Secondary at {secondary_idx}, "
            f"Confidence at {confidence_idx}"
        )

    def test_no_restaurant_metadata_is_missing_any_tag(self, df):
        """
        Comprehensive sweep: every single restaurant must have all 3
        atmosphere tags present simultaneously — partial enrichment
        would indicate the pipeline crashed mid-row.
        """
        incomplete = []
        for _, row in df.iterrows():
            meta = row["restaurant_metadata"]
            has_primary    = "Primary Atmosphere:" in meta
            has_secondary  = "Secondary Atmosphere:" in meta
            has_confidence = "Atmosphere Confidence:" in meta
            if not (has_primary and has_secondary and has_confidence):
                incomplete.append(row["Name"])
        assert not incomplete, (
            f"Restaurants with incomplete atmosphere tags: {incomplete}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. WRAPPER UNIT TESTS — classify_atmosphere
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyAtmosphere:

    def test_returns_top_label_as_predicted_atmosphere(self):
        """classify_atmosphere must return the highest-scored label as predicted."""
        mock_classifier = build_atmosphere_mock(make_fine_dining_dominant_scores())
        predicted, _, _ = classify_atmosphere(mock_classifier, "an elegant fine dining room")
        assert predicted == "Fine Dining / Formal"

    def test_returns_second_label_as_secondary_atmosphere(self):
        """classify_atmosphere must return the second-highest label as secondary."""
        mock_classifier = build_atmosphere_mock(make_fine_dining_dominant_scores())
        _, _, secondary = classify_atmosphere(mock_classifier, "an elegant fine dining room")
        assert secondary == "Upscale Casual"

    def test_returns_correct_confidence_score(self):
        """classify_atmosphere must return the score matching the top label."""
        mock_classifier = build_atmosphere_mock(make_romantic_dominant_scores())
        _, confidence, _ = classify_atmosphere(mock_classifier, "candlelit dinner for two")
        assert confidence == 0.9512

    def test_pipeline_called_with_multi_label_true(self):
        """
        classify_atmosphere must call the pipeline with multi_label=True.
        This ensures each atmosphere label gets an independent probability
        rather than a forced single-class softmax — necessary because a
        restaurant can genuinely be both Romantic and Cozy/Intimate.
        """
        mock_classifier = MagicMock()
        mock_classifier.return_value = {
            "sequence": "test",
            "labels": ["Trendy / Hip", "Industrial / Urban"],
            "scores": [0.7, 0.5],
        }
        classify_atmosphere(mock_classifier, "industrial chic wine bar")
        _, kwargs = mock_classifier.call_args
        assert kwargs.get("multi_label") is True, (
            f"multi_label was not set to True. kwargs passed: {kwargs}"
        )

    def test_pipeline_called_with_all_thirteen_candidate_labels(self):
        """
        classify_atmosphere must call the pipeline with the full 13-label
        candidate set — not a subset, not a superset.
        """
        mock_classifier = MagicMock()
        mock_classifier.return_value = {
            "sequence": "test",
            "labels": ["Casual", "Trendy / Hip"],
            "scores": [0.6, 0.4],
        }
        classify_atmosphere(mock_classifier, "a relaxed neighborhood spot")
        args, kwargs = mock_classifier.call_args
        passed_candidates = kwargs.get("candidate_labels", args[1] if len(args) > 1 else None)
        assert passed_candidates is not None, "candidate_labels not passed to pipeline"
        assert set(passed_candidates) == set(ATMOSPHERE_LABELS), (
            f"Pipeline called with wrong candidate labels.\n"
            f"Expected: {sorted(ATMOSPHERE_LABELS)}\n"
            f"Got:      {sorted(passed_candidates)}"
        )

    @pytest.mark.parametrize("label", ATMOSPHERE_LABELS)
    def test_every_valid_atmosphere_label_can_be_returned_as_primary(self, label):
        """
        Any of the 13 valid atmosphere labels can be the top result.
        The wrapper must not filter or remap any of them.
        """
        scores = {l: 0.1 for l in ATMOSPHERE_LABELS}
        scores[label] = 0.95
        mock_classifier = build_atmosphere_mock(scores)
        predicted, _, _ = classify_atmosphere(mock_classifier, "restaurant description")
        assert predicted == label, (
            f"Wrapper remapped label '{label}' to '{predicted}'"
        )

    def test_confidence_rounded_to_four_decimal_places(self):
        """Confidence score must be rounded to 4 decimal places."""
        scores = {label: 0.01 for label in ATMOSPHERE_LABELS}
        scores["Trendy / Hip"] = 0.123456789
        mock_classifier = build_atmosphere_mock(scores)
        _, confidence, _ = classify_atmosphere(mock_classifier, "hip wine bar")
        assert confidence == round(0.123456789, 4)

    def test_pipeline_called_exactly_once(self):
        """classify_atmosphere must call the pipeline exactly once per invocation."""
        mock_classifier = build_atmosphere_mock(make_fine_dining_dominant_scores())
        classify_atmosphere(mock_classifier, "fine dining establishment")
        assert mock_classifier.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 8. WRAPPER UNIT TESTS — append_atmosphere_tags
# ═══════════════════════════════════════════════════════════════════════════════

class TestAppendAtmosphereTags:

    def test_appends_all_three_tags(self):
        """append_atmosphere_tags must append Primary, Secondary, and Confidence."""
        base_metadata = "Test Bistro is a French restaurant. Price range: $$$$."
        result = append_atmosphere_tags(
            base_metadata, "Romantic", 0.9512, "Cozy / Intimate"
        )
        assert "Primary Atmosphere: Romantic." in result
        assert "Secondary Atmosphere: Cozy / Intimate." in result
        assert "Atmosphere Confidence: 0.9512." in result

    def test_original_metadata_preserved_at_start(self):
        """The original metadata text must remain untouched at the start of the string."""
        base_metadata = "Somni is a Spanish Modernist restaurant in West Hollywood."
        result = append_atmosphere_tags(base_metadata, "Fine Dining / Formal", 0.99, "Traditional / Classic")
        assert result.startswith(base_metadata)

    def test_tags_appended_in_correct_order(self):
        """Tags must appear Primary -> Secondary -> Confidence, in that order."""
        result = append_atmosphere_tags("Base text.", "Casual", 0.75, "Trendy / Hip")
        primary_idx    = result.find("Primary Atmosphere:")
        secondary_idx  = result.find("Secondary Atmosphere:")
        confidence_idx = result.find("Atmosphere Confidence:")
        assert primary_idx < secondary_idx < confidence_idx

    def test_result_is_longer_than_input(self):
        """The enriched metadata must always be longer than the input metadata."""
        base_metadata = "A simple restaurant description."
        result = append_atmosphere_tags(base_metadata, "Upscale Casual", 0.81, "Fine Casual")
        assert len(result) > len(base_metadata)

    def test_special_characters_in_atmosphere_label_preserved(self):
        """
        Atmosphere labels containing slashes (e.g. 'Fine Dining / Formal')
        must be preserved exactly — no escaping or truncation at the slash.
        """
        result = append_atmosphere_tags("Base.", "Fine Dining / Formal", 0.95, "Theatrical / Entertainment")
        assert "Primary Atmosphere: Fine Dining / Formal." in result
        assert "Secondary Atmosphere: Theatrical / Entertainment." in result


# ═══════════════════════════════════════════════════════════════════════════════
# 9. EDGE CASE TESTS ON THE WRAPPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassificationEdgeCases:

    def test_very_low_confidence_score_still_returned(self):
        """
        A near-zero confidence score (e.g., 0.0001) must still be
        returned — the wrapper must not suppress low-confidence results.
        """
        scores = {label: 0.0001 for label in ATMOSPHERE_LABELS}
        scores["Casual"] = 0.0002  # still technically the top label
        mock_classifier = build_atmosphere_mock(scores)
        predicted, confidence, _ = classify_atmosphere(mock_classifier, "ambiguous restaurant")
        assert predicted == "Casual"
        assert confidence > 0.0

    def test_perfect_confidence_score_still_returned(self):
        """A perfect 1.0 confidence score must pass through unchanged."""
        scores = make_fine_dining_dominant_scores(**{"Fine Dining / Formal": 1.0})
        mock_classifier = build_atmosphere_mock(scores)
        predicted, confidence, _ = classify_atmosphere(mock_classifier, "the finest dining room")
        assert predicted == "Fine Dining / Formal"
        assert confidence == 1.0

    def test_empty_string_input_does_not_raise(self):
        """
        An empty string input must not raise. The pipeline mock will return
        a valid result regardless of the input text.
        """
        mock_classifier = build_atmosphere_mock(make_trendy_dominant_scores())
        try:
            classify_atmosphere(mock_classifier, "")
        except Exception as e:
            pytest.fail(f"classify_atmosphere raised on empty string: {e}")

    def test_pipeline_input_text_is_passed_correctly(self):
        """
        The input text passed to classify_atmosphere must be forwarded
        verbatim to the pipeline — the wrapper must not modify it.
        """
        mock_classifier = MagicMock()
        mock_classifier.return_value = {
            "sequence": "test",
            "labels": ["Romantic", "Cozy / Intimate"],
            "scores": [0.8, 0.6],
        }
        test_text = "A 14-seat candlelit dining room with dim lighting."
        classify_atmosphere(mock_classifier, test_text)
        args, _ = mock_classifier.call_args
        assert args[0] == test_text, (
            f"Pipeline received '{args[0]}' instead of '{test_text}'"
        )

    def test_confidence_is_numeric_not_string(self):
        """The confidence value returned by classify_atmosphere must be a float."""
        mock_classifier = build_atmosphere_mock(make_romantic_dominant_scores())
        _, confidence, _ = classify_atmosphere(mock_classifier, "romantic dinner spot")
        assert isinstance(confidence, float), (
            f"Expected float confidence, got {type(confidence)}"
        )

    def test_special_characters_in_text_do_not_raise(self):
        """
        Restaurant metadata often contains apostrophes, slashes, quotes,
        and ampersands. The wrapper must not raise on any of these.
        """
        mock_classifier = build_atmosphere_mock(make_fine_dining_dominant_scores())
        tricky_texts = [
            "Chef's Table — omakase & kaiseki",
            "n/naka: 13-course tasting menu",
            'Described as "exceptional" by Michelin',
            "Gwen Butcher Shop & Restaurant",
            "Pasta | Bar — Italian-inspired tasting menu",
        ]
        for text in tricky_texts:
            try:
                classify_atmosphere(mock_classifier, text)
            except Exception as e:
                pytest.fail(
                    f"classify_atmosphere raised on text with special chars: {e}\n"
                    f"Text: {text}"
                )

    def test_two_label_minimum_response_does_not_raise(self):
        """
        If the pipeline is somehow called with only the minimum 2 labels
        needed for primary + secondary, classify_atmosphere must not raise
        an IndexError when extracting secondary_atmosphere.
        """
        mock_classifier = MagicMock()
        mock_classifier.return_value = {
            "sequence": "test",
            "labels":  ["Cozy / Intimate", "Romantic"],
            "scores":  [0.7, 0.5],
        }
        try:
            predicted, confidence, secondary = classify_atmosphere(mock_classifier, "small cozy spot")
        except IndexError as e:
            pytest.fail(f"classify_atmosphere raised IndexError on 2-label response: {e}")
        assert predicted == "Cozy / Intimate"
        assert secondary == "Romantic"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MOCK INTEGRITY TESTS
# Ensures that the mock infrastructure is working correctly and that
# no test accidentally triggers a real HuggingFace model download.
# ═══════════════════════════════════════════════════════════════════════════════

class TestMockIntegrity:

    def test_mock_classifier_is_callable(self):
        """The mock returned by build_atmosphere_mock must be callable."""
        mock_classifier = build_atmosphere_mock(make_fine_dining_dominant_scores())
        assert callable(mock_classifier)

    def test_mock_returns_correct_output_structure(self):
        """The mock's return value must match the real pipeline's output shape."""
        mock_classifier = build_atmosphere_mock(make_romantic_dominant_scores())
        result = mock_classifier("any text", candidate_labels=ATMOSPHERE_LABELS, multi_label=True)
        assert "sequence" in result
        assert "labels" in result
        assert "scores" in result
        assert result["labels"][0] == "Romantic"
        assert result["scores"][0] == 0.9512

    def test_mock_label_with_highest_score_is_always_first(self):
        """
        The dominant label must always be first in the returned labels list,
        matching how the real zero-shot pipeline sorts results descending.
        """
        for scores_fn in [make_fine_dining_dominant_scores, make_romantic_dominant_scores, make_trendy_dominant_scores]:
            mock_classifier = build_atmosphere_mock(scores_fn())
            result = mock_classifier("text", candidate_labels=ATMOSPHERE_LABELS, multi_label=True)
            top_label = result["labels"][0]
            top_score = result["scores"][0]
            assert top_score == max(result["scores"]), (
                f"Top label '{top_label}' does not have the highest score"
            )

    def test_mock_scores_sorted_descending(self):
        """
        The mock must return scores in descending order, consistent with
        the real zero-shot pipeline's output format.
        """
        mock_classifier = build_atmosphere_mock(make_fine_dining_dominant_scores())
        result = mock_classifier("text", candidate_labels=ATMOSPHERE_LABELS, multi_label=True)
        scores = result["scores"]
        assert scores == sorted(scores, reverse=True), (
            f"Mock scores are not sorted descending: {scores}"
        )

    def test_mock_includes_all_thirteen_labels(self):
        """The mock must return a score for all 13 atmosphere labels, not a subset."""
        mock_classifier = build_atmosphere_mock(make_trendy_dominant_scores())
        result = mock_classifier("text", candidate_labels=ATMOSPHERE_LABELS, multi_label=True)
        assert len(result["labels"]) == 13, (
            f"Expected 13 labels in mock output, got {len(result['labels'])}"
        )
        assert set(result["labels"]) == set(ATMOSPHERE_LABELS)

    def test_wrapper_does_not_call_pipeline_more_than_once(self):
        """
        classify_atmosphere must call the pipeline exactly once. Multiple
        calls would indicate the wrapper is retrying or looping unexpectedly.
        """
        mock_classifier = build_atmosphere_mock(make_romantic_dominant_scores())
        classify_atmosphere(mock_classifier, "restaurant text")
        assert mock_classifier.call_count == 1, (
            f"Pipeline called {mock_classifier.call_count} times (expected 1)"
        )
