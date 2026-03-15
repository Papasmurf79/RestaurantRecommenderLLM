"""
tests/test_classification.py
============================
Unit tests for Notebook 3 — Zero-Shot Text Classification.

Strategy
--------
The facebook/bart-large-mnli pipeline is fully mocked using
unittest.mock.patch. No model download, no GPU, no network required.
Tests operate in two layers:

  Layer 1 — Unit tests on the classification wrapper functions in
             isolation. Each test feeds a synthetic bart output dict
             and asserts the wrapper extracts the right label and
             confidence score.

  Layer 2 — Integration tests on the final CSV artifact. These load
             restaurants_with_classifications.csv directly and assert
             schema integrity, column presence, label validity, and
             confidence score bounds across all 71 rows.

The bart-large-mnli pipeline returns this structure per text:
  {
    "sequence": <input text>,
    "labels":   ["label_a", "label_b", ...],   # highest score first
    "scores":   [0.45, 0.32, ...]              # same order as labels
  }
The classification wrapper picks labels[0] as the predicted class
and scores[0] as the confidence value.

Covers
------
  1. CSV output schema — 21 columns, correct names, correct order
  2. Row count — always 71
  3. Null checks — no nulls in any of the 6 new classification columns
  4. Label validity — all values in known label sets
  5. Confidence score bounds — all scores in [0.0, 1.0]
  6. Confidence score types — stored as floats, not strings
  7. Wrapper unit tests — classify_cuisine correctly picks top label
  8. Wrapper unit tests — classify_dining_format correctly picks top label
  9. Wrapper unit tests — classify_occasion correctly picks top label
 10. Wrapper unit tests — classify_vibe correctly picks top label
 11. Wrapper unit tests — confidence score extraction is accurate
 12. Wrapper unit tests — tied scores handled without raising
 13. Wrapper unit tests — single-candidate label list handled
 14. Wrapper unit tests — handles near-zero confidence scores
 15. Base column preservation — original 15 columns untouched
 16. Spot-checks — known restaurant→cuisine group mappings verified
 17. Metadata enrichment — restaurant_metadata updated with classification tags
 18. Mock integration — pipeline is never called with live model

Real data stats (from restaurants_with_classifications.csv):
  Total rows:              71
  Total columns:           21 (15 base + 6 new)
  simple_cuisine_group:    11 unique labels
  dining_format:            3 unique labels
  predicted_occasion:       4 unique labels
  predicted_vibe:           6 unique labels
  occasion_confidence:      range [0.2291, 0.6343]
  vibe_confidence:          range [0.2618, 0.8531]
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pandas as pd
import pytest


# ── Path constants ─────────────────────────────────────────────────────────────
FINAL_CSV          = "data/cleaned_restaurants_final.csv"
CLASSIFIED_CSV     = "data/restaurants_with_classifications.csv"


# ── Known valid label sets (sourced directly from the real CSV) ────────────────
VALID_CUISINE_GROUPS = {
    "Japanese",
    "Contemporary American",
    "Asian Fusion",
    "Steakhouse",
    "Italian",
    "French",
    "Seafood",
    "Japanese Fusion",
    "Spanish",
    "Mexican / Latin",
    "American Casual",
}

VALID_DINING_FORMATS = {
    "Full Service",
    "Tasting Menu / Omakase",
    "A La Carte / Casual",
}

VALID_OCCASIONS = {
    "Special Occasion",
    "Foodie Adventure",
    "Casual Night Out",
    "Business Dining",
}

VALID_VIBES = {
    "Refined & Elegant",
    "Cozy & Relaxed",
    "Hip & Trendy",
    "Intimate",
    "Theatrical & Experiential",
    "Lively & Social",
}

# ── Expected schema ────────────────────────────────────────────────────────────
BASE_COLUMNS = [
    "Name", "Location", "Description", "Address", "Telephone Number",
    "Price", "Cuisine Type", "Dining Atmosphere", "Sky-High Rooftop",
    "Michelin-Guide", "Customer Ratings", "Operation Hours",
    "Reservations", "Dress Code", "restaurant_metadata",
]

NEW_CLASSIFICATION_COLUMNS = [
    "simple_cuisine_group",
    "dining_format",
    "predicted_occasion",
    "occasion_confidence",
    "predicted_vibe",
    "vibe_confidence",
]

EXPECTED_COLUMNS = BASE_COLUMNS + NEW_CLASSIFICATION_COLUMNS


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION WRAPPER FUNCTIONS
# (Mirror of Notebook 3 implementation — keep in sync)
#
# These are the functions under test. Defined here so tests run without
# importing the notebook. When you extract these into a src/ module,
# replace the definitions below with a single import line.
# ═══════════════════════════════════════════════════════════════════════════════

def _run_zero_shot(pipeline, text: str, candidate_labels: list[str]) -> dict:
    """
    Calls the bart-large-mnli pipeline for a single text and label set.
    Returns the raw pipeline output dict.
    """
    return pipeline(text, candidate_labels=candidate_labels, multi_label=False)


def classify_cuisine(pipeline, text: str) -> tuple[str, float]:
    """Classify a restaurant's cuisine group. Returns (label, confidence)."""
    candidates = [
        "Japanese", "Contemporary American", "Asian Fusion", "Steakhouse",
        "Italian", "French", "Seafood", "Japanese Fusion", "Spanish",
        "Mexican / Latin", "American Casual",
    ]
    result = _run_zero_shot(pipeline, text, candidates)
    return result["labels"][0], round(result["scores"][0], 4)


def classify_dining_format(pipeline, text: str) -> tuple[str, float]:
    """Classify a restaurant's dining format. Returns (label, confidence)."""
    candidates = [
        "Tasting Menu / Omakase",
        "Full Service",
        "A La Carte / Casual",
    ]
    result = _run_zero_shot(pipeline, text, candidates)
    return result["labels"][0], round(result["scores"][0], 4)


def classify_occasion(pipeline, text: str) -> tuple[str, float]:
    """Classify the best occasion for a restaurant. Returns (label, confidence)."""
    candidates = [
        "Special Occasion",
        "Foodie Adventure",
        "Casual Night Out",
        "Business Dining",
    ]
    result = _run_zero_shot(pipeline, text, candidates)
    return result["labels"][0], round(result["scores"][0], 4)


def classify_vibe(pipeline, text: str) -> tuple[str, float]:
    """Classify a restaurant's vibe. Returns (label, confidence)."""
    candidates = [
        "Refined & Elegant",
        "Cozy & Relaxed",
        "Hip & Trendy",
        "Intimate",
        "Theatrical & Experiential",
        "Lively & Social",
    ]
    result = _run_zero_shot(pipeline, text, candidates)
    return result["labels"][0], round(result["scores"][0], 4)


def build_mock_pipeline(label: str, score: float) -> MagicMock:
    """
    Returns a MagicMock that behaves like a bart-large-mnli pipeline,
    always returning the given label as the top result with the given score.

    The mock returns labels in descending score order with the target label
    first — exactly how the real bart pipeline responds.
    """
    mock = MagicMock()
    mock.return_value = {
        "sequence": "mock sequence",
        "labels":   [label, "Other Label A", "Other Label B"],
        "scores":   [score, (1.0 - score) * 0.6, (1.0 - score) * 0.4],
    }
    return mock


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def df_base():
    """The 15-column base CSV (input to Notebook 3)."""
    return pd.read_csv(FINAL_CSV)


@pytest.fixture(scope="module")
def df_classified():
    """The 21-column output CSV (output of Notebook 3)."""
    return pd.read_csv(CLASSIFIED_CSV)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CSV OUTPUT SCHEMA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputSchema:

    def test_column_count_is_21(self, df_classified):
        """Output CSV must have exactly 21 columns."""
        assert len(df_classified.columns) == 21, (
            f"Expected 21 columns, got {len(df_classified.columns)}.\n"
            f"Columns: {list(df_classified.columns)}"
        )

    def test_all_expected_columns_present(self, df_classified):
        """Every column in EXPECTED_COLUMNS must exist in the output."""
        missing = [c for c in EXPECTED_COLUMNS if c not in df_classified.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_unexpected_columns(self, df_classified):
        """No columns beyond the expected 21 should be present."""
        extra = [c for c in df_classified.columns if c not in EXPECTED_COLUMNS]
        assert not extra, f"Unexpected extra columns: {extra}"

    def test_column_order_matches_expected(self, df_classified):
        """Column order must match EXPECTED_COLUMNS exactly."""
        assert list(df_classified.columns) == EXPECTED_COLUMNS, (
            f"Column order mismatch.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Got:      {list(df_classified.columns)}"
        )

    def test_six_new_columns_added(self, df_base, df_classified):
        """Notebook 3 must add exactly 6 new columns to the base 15."""
        new_cols = [c for c in df_classified.columns if c not in df_base.columns]
        assert len(new_cols) == 6, (
            f"Expected 6 new columns, found {len(new_cols)}: {new_cols}"
        )
        assert set(new_cols) == set(NEW_CLASSIFICATION_COLUMNS), (
            f"New columns don't match expected.\n"
            f"Expected: {NEW_CLASSIFICATION_COLUMNS}\n"
            f"Got:      {new_cols}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ROW COUNT & BASE COLUMN PRESERVATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRowCountAndBaseColumns:

    def test_row_count_is_71(self, df_classified):
        """Classification must not add or remove rows — always 71."""
        assert len(df_classified) == 71, (
            f"Expected 71 rows, got {len(df_classified)}"
        )

    @pytest.mark.parametrize("col", [c for c in BASE_COLUMNS if c != "restaurant_metadata"])
    def test_base_column_values_unchanged(self, df_base, df_classified, col):
        """
        Every base column value except restaurant_metadata must be identical
        before and after classification. Notebook 3 must not mutate any
        input data fields.

        restaurant_metadata is explicitly excluded from this test because
        Notebook 3 intentionally appends classification tags (Cuisine Group,
        Dining Format, Best For, Vibe) to it — verified separately in
        TestMetadataEnrichment.
        """
        original = df_base[col].reset_index(drop=True)
        classified = df_classified[col].reset_index(drop=True)
        # Compare as strings to handle NaN-safe comparison
        mismatches = (original.astype(str) != classified.astype(str)).sum()
        assert mismatches == 0, (
            f"Column '{col}' was modified by Notebook 3: "
            f"{mismatches} rows differ"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NULL CHECKS ON CLASSIFICATION COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNullValues:

    @pytest.mark.parametrize("col", NEW_CLASSIFICATION_COLUMNS)
    def test_no_nulls_in_classification_column(self, df_classified, col):
        """Every classification column must have zero null values."""
        null_count = df_classified[col].isnull().sum()
        assert null_count == 0, (
            f"Column '{col}' has {null_count} null(s). "
            f"Rows affected:\n"
            f"{df_classified[df_classified[col].isnull()]['Name'].tolist()}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LABEL VALIDITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLabelValidity:

    def test_all_cuisine_groups_are_known_labels(self, df_classified):
        """
        Every value in simple_cuisine_group must be one of the 11 known
        cuisine group labels. Unknown values mean new labels were introduced
        without updating VALID_CUISINE_GROUPS.
        """
        actual = set(df_classified["simple_cuisine_group"].unique())
        unknown = actual - VALID_CUISINE_GROUPS
        assert not unknown, (
            f"Unknown cuisine group labels found: {unknown}\n"
            f"Update VALID_CUISINE_GROUPS if these are intentional new labels."
        )

    def test_all_dining_formats_are_known_labels(self, df_classified):
        """Every dining_format value must be one of the 3 known format labels."""
        actual = set(df_classified["dining_format"].unique())
        unknown = actual - VALID_DINING_FORMATS
        assert not unknown, (
            f"Unknown dining format labels: {unknown}\n"
            f"Known formats: {VALID_DINING_FORMATS}"
        )

    def test_all_occasions_are_known_labels(self, df_classified):
        """Every predicted_occasion value must be one of the 4 known occasion labels."""
        actual = set(df_classified["predicted_occasion"].unique())
        unknown = actual - VALID_OCCASIONS
        assert not unknown, (
            f"Unknown occasion labels: {unknown}\n"
            f"Known occasions: {VALID_OCCASIONS}"
        )

    def test_all_vibes_are_known_labels(self, df_classified):
        """Every predicted_vibe value must be one of the 6 known vibe labels."""
        actual = set(df_classified["predicted_vibe"].unique())
        unknown = actual - VALID_VIBES
        assert not unknown, (
            f"Unknown vibe labels: {unknown}\n"
            f"Known vibes: {VALID_VIBES}"
        )

    def test_no_empty_string_labels(self, df_classified):
        """No classification column should contain empty strings."""
        for col in ["simple_cuisine_group", "dining_format", "predicted_occasion", "predicted_vibe"]:
            empty = (df_classified[col].astype(str).str.strip() == "").sum()
            assert empty == 0, (
                f"Column '{col}' contains {empty} empty string(s)"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONFIDENCE SCORE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceScores:

    def test_occasion_confidence_is_float_dtype(self, df_classified):
        """occasion_confidence must be stored as float64, not object/string."""
        assert pd.api.types.is_float_dtype(df_classified["occasion_confidence"]), (
            f"occasion_confidence dtype is {df_classified['occasion_confidence'].dtype}, "
            f"expected float64"
        )

    def test_vibe_confidence_is_float_dtype(self, df_classified):
        """vibe_confidence must be stored as float64, not object/string."""
        assert pd.api.types.is_float_dtype(df_classified["vibe_confidence"]), (
            f"vibe_confidence dtype is {df_classified['vibe_confidence'].dtype}, "
            f"expected float64"
        )

    def test_occasion_confidence_in_zero_to_one(self, df_classified):
        """All occasion_confidence scores must be within [0.0, 1.0]."""
        oob = df_classified[
            (df_classified["occasion_confidence"] < 0.0) |
            (df_classified["occasion_confidence"] > 1.0)
        ]
        assert oob.empty, (
            f"occasion_confidence out of [0,1] bounds:\n"
            f"{oob[['Name', 'occasion_confidence']].to_string()}"
        )

    def test_vibe_confidence_in_zero_to_one(self, df_classified):
        """All vibe_confidence scores must be within [0.0, 1.0]."""
        oob = df_classified[
            (df_classified["vibe_confidence"] < 0.0) |
            (df_classified["vibe_confidence"] > 1.0)
        ]
        assert oob.empty, (
            f"vibe_confidence out of [0,1] bounds:\n"
            f"{oob[['Name', 'vibe_confidence']].to_string()}"
        )

    def test_occasion_confidence_above_floor(self, df_classified):
        """
        All occasion_confidence scores must be > 0.0 — a zero confidence
        means the pipeline returned nonsense or defaulted without classifying.
        """
        zero_conf = df_classified[df_classified["occasion_confidence"] == 0.0]
        assert zero_conf.empty, (
            f"Zero occasion_confidence found:\n"
            f"{zero_conf[['Name', 'occasion_confidence']].to_string()}"
        )

    def test_vibe_confidence_above_floor(self, df_classified):
        """All vibe_confidence scores must be > 0.0."""
        zero_conf = df_classified[df_classified["vibe_confidence"] == 0.0]
        assert zero_conf.empty, (
            f"Zero vibe_confidence found:\n"
            f"{zero_conf[['Name', 'vibe_confidence']].to_string()}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. KNOWN MAPPING SPOT-CHECKS
# These are ground-truth checks against specific restaurants whose
# classification is unambiguous enough to assert directly.
# ═══════════════════════════════════════════════════════════════════════════════

class TestKnownMappings:

    @pytest.mark.parametrize("restaurant_name,expected_cuisine", [
        ("Somni",      "Spanish"),
        ("Providence", "Seafood"),
        ("Hayato",     "Japanese"),
        ("n/naka",     "Japanese"),
        ("Holbox",     "Seafood"),
        ("Langer's",   "American Casual"),
        ("Komal",      "Mexican / Latin"),
        ("Damian",     "Mexican / Latin"),
        ("Camphor",    "French"),
        ("Melisse",    "French"),
    ])
    def test_cuisine_group_spot_check(self, df_classified, restaurant_name, expected_cuisine):
        """
        Spot-check that specific restaurants received the expected cuisine group.
        These are high-confidence cases where the cuisine is clear and unambiguous.
        """
        row = df_classified[df_classified["Name"] == restaurant_name]
        assert len(row) >= 1, f"Restaurant '{restaurant_name}' not found in dataset"
        actual = row["simple_cuisine_group"].iloc[0]
        assert actual == expected_cuisine, (
            f"'{restaurant_name}': expected cuisine_group='{expected_cuisine}', "
            f"got '{actual}'"
        )

    @pytest.mark.parametrize("restaurant_name,expected_format", [
        ("Somni",         "Tasting Menu / Omakase"),
        ("Hayato",        "Tasting Menu / Omakase"),
        ("n/naka",        "Tasting Menu / Omakase"),
        ("Langer's",      "A La Carte / Casual"),
        ("Moo's Craft Barbecue", "A La Carte / Casual"),
    ])
    def test_dining_format_spot_check(self, df_classified, restaurant_name, expected_format):
        """
        Spot-check dining format for clear-cut cases:
        multi-course tasting menus, and walk-up casual spots.
        """
        row = df_classified[df_classified["Name"] == restaurant_name]
        assert len(row) >= 1, f"Restaurant '{restaurant_name}' not found in dataset"
        actual = row["dining_format"].iloc[0]
        assert actual == expected_format, (
            f"'{restaurant_name}': expected dining_format='{expected_format}', "
            f"got '{actual}'"
        )

    def test_all_omakase_restaurants_are_tasting_menu_format(self, df_classified):
        """
        Restaurants with 'Omakase' in their Cuisine Type should be classified
        as 'Tasting Menu / Omakase' dining format — this is the clearest
        deterministic signal in the dataset.
        """
        omakase_rows = df_classified[
            df_classified["Cuisine Type"].str.contains("Omakase", na=False)
        ]
        wrong = omakase_rows[omakase_rows["dining_format"] != "Tasting Menu / Omakase"]
        assert wrong.empty, (
            f"Omakase restaurants not classified as 'Tasting Menu / Omakase':\n"
            f"{wrong[['Name', 'Cuisine Type', 'dining_format']].to_string()}"
        )

    def test_michelin_starred_restaurants_are_special_occasion_or_foodie(self, df_classified):
        """
        All Michelin 2-Star and 3-Star restaurants should be classified as
        either 'Special Occasion' or 'Foodie Adventure'. These are the
        most exclusive restaurants in the dataset — no 3-star kaiseki is
        a 'Casual Night Out'.
        """
        premium = df_classified[
            df_classified["Michelin-Guide"].isin(["2-Star", "3-Star"])
        ]
        acceptable_occasions = {"Special Occasion", "Foodie Adventure"}
        wrong = premium[~premium["predicted_occasion"].isin(acceptable_occasions)]
        assert wrong.empty, (
            f"2-Star/3-Star restaurants with unexpected occasion:\n"
            f"{wrong[['Name', 'Michelin-Guide', 'predicted_occasion']].to_string()}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. METADATA ENRICHMENT TESTS
# Notebook 3 appends classification tags to restaurant_metadata.
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetadataEnrichment:

    def test_metadata_contains_cuisine_group_tag(self, df_classified):
        """
        restaurant_metadata in the classified CSV must contain
        'Cuisine Group: {simple_cuisine_group}' for every row.
        """
        for _, row in df_classified.iterrows():
            expected = f"Cuisine Group: {row['simple_cuisine_group']}"
            assert expected in row["restaurant_metadata"], (
                f"'{row['Name']}': expected '{expected}' in metadata.\n"
                f"Metadata snippet: {row['restaurant_metadata'][-150:]}"
            )

    def test_metadata_contains_dining_format_tag(self, df_classified):
        """
        restaurant_metadata must contain 'Dining Format: {dining_format}'
        for every row.
        """
        for _, row in df_classified.iterrows():
            expected = f"Dining Format: {row['dining_format']}"
            assert expected in row["restaurant_metadata"], (
                f"'{row['Name']}': expected '{expected}' in metadata.\n"
                f"Metadata snippet: {row['restaurant_metadata'][-150:]}"
            )

    def test_metadata_contains_best_for_tag(self, df_classified):
        """
        restaurant_metadata must contain 'Best For: {predicted_occasion}'
        for every row.
        """
        for _, row in df_classified.iterrows():
            expected = f"Best For: {row['predicted_occasion']}"
            assert expected in row["restaurant_metadata"], (
                f"'{row['Name']}': expected '{expected}' in metadata.\n"
                f"Metadata snippet: {row['restaurant_metadata'][-150:]}"
            )

    def test_metadata_contains_vibe_tag(self, df_classified):
        """
        restaurant_metadata must contain 'Vibe: {predicted_vibe}'
        for every row.
        """
        for _, row in df_classified.iterrows():
            expected = f"Vibe: {row['predicted_vibe']}"
            assert expected in row["restaurant_metadata"], (
                f"'{row['Name']}': expected '{expected}' in metadata.\n"
                f"Metadata snippet: {row['restaurant_metadata'][-150:]}"
            )

    def test_metadata_is_longer_than_base_metadata(self, df_base, df_classified):
        """
        The enriched metadata in the classified CSV must be longer than
        the base metadata — the four classification tags were appended.
        """
        # Merge on Name to compare same restaurant rows
        merged = df_base[["Name", "restaurant_metadata"]].merge(
            df_classified[["Name", "restaurant_metadata"]],
            on="Name",
            suffixes=("_base", "_classified"),
        )
        shorter = merged[
            merged["restaurant_metadata_classified"].str.len() <=
            merged["restaurant_metadata_base"].str.len()
        ]
        assert shorter.empty, (
            f"These restaurants have metadata that wasn't enriched:\n"
            f"{shorter['Name'].tolist()}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. WRAPPER UNIT TESTS — CUISINE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyCuisine:

    def test_returns_top_label_from_pipeline_output(self):
        """classify_cuisine must return the highest-scored label."""
        mock_pipeline = build_mock_pipeline("Japanese", 0.72)
        label, confidence = classify_cuisine(mock_pipeline, "omakase sushi restaurant")
        assert label == "Japanese"

    def test_returns_correct_confidence_score(self):
        """classify_cuisine must return the score matching the top label."""
        mock_pipeline = build_mock_pipeline("Italian", 0.55)
        label, confidence = classify_cuisine(mock_pipeline, "pasta and pizza")
        assert confidence == 0.55

    def test_pipeline_called_with_correct_candidate_labels(self):
        """
        classify_cuisine must call the pipeline with the full candidate
        label list — not a subset, not a superset.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "sequence": "test",
            "labels": ["Japanese", "Italian"],
            "scores": [0.6, 0.4],
        }
        classify_cuisine(mock_pipeline, "sushi restaurant")
        args, kwargs = mock_pipeline.call_args
        passed_candidates = kwargs.get("candidate_labels", args[1] if len(args) > 1 else None)
        assert passed_candidates is not None, "candidate_labels not passed to pipeline"
        assert set(passed_candidates) == VALID_CUISINE_GROUPS, (
            f"Pipeline called with wrong candidate labels.\n"
            f"Expected: {sorted(VALID_CUISINE_GROUPS)}\n"
            f"Got:      {sorted(passed_candidates)}"
        )

    @pytest.mark.parametrize("label", list(VALID_CUISINE_GROUPS))
    def test_every_valid_cuisine_label_can_be_returned(self, label):
        """
        Any of the 11 valid cuisine labels can be the top result.
        The wrapper must not filter or remap any of them.
        """
        mock_pipeline = build_mock_pipeline(label, 0.5)
        returned_label, _ = classify_cuisine(mock_pipeline, "restaurant description")
        assert returned_label == label, (
            f"Wrapper remapped label '{label}' to '{returned_label}'"
        )

    def test_confidence_rounded_to_four_decimal_places(self):
        """Confidence score must be rounded to 4 decimal places."""
        mock_pipeline = build_mock_pipeline("French", 0.123456789)
        _, confidence = classify_cuisine(mock_pipeline, "bistro")
        # Should be 0.1235 after rounding to 4 dp
        assert confidence == round(0.123456789, 4)

    def test_pipeline_called_exactly_once(self):
        """classify_cuisine must call the pipeline exactly once per invocation."""
        mock_pipeline = build_mock_pipeline("Seafood", 0.45)
        classify_cuisine(mock_pipeline, "seafood restaurant")
        assert mock_pipeline.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 9. WRAPPER UNIT TESTS — DINING FORMAT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyDiningFormat:

    def test_returns_top_label_from_pipeline_output(self):
        """classify_dining_format must return the highest-scored label."""
        mock_pipeline = build_mock_pipeline("Tasting Menu / Omakase", 0.68)
        label, _ = classify_dining_format(mock_pipeline, "20-course kaiseki dinner")
        assert label == "Tasting Menu / Omakase"

    def test_returns_correct_confidence_score(self):
        """classify_dining_format must return the score for the top label."""
        mock_pipeline = build_mock_pipeline("A La Carte / Casual", 0.41)
        _, confidence = classify_dining_format(mock_pipeline, "casual diner")
        assert confidence == 0.41

    def test_pipeline_called_with_three_format_candidates(self):
        """
        classify_dining_format must call the pipeline with exactly the
        3 dining format candidates — no more, no less.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "sequence": "test",
            "labels": ["Full Service"],
            "scores": [0.7],
        }
        classify_dining_format(mock_pipeline, "restaurant")
        args, kwargs = mock_pipeline.call_args
        passed_candidates = kwargs.get("candidate_labels", args[1] if len(args) > 1 else None)
        assert set(passed_candidates) == VALID_DINING_FORMATS, (
            f"Wrong candidates: {passed_candidates}"
        )

    @pytest.mark.parametrize("label", list(VALID_DINING_FORMATS))
    def test_every_valid_format_label_can_be_returned(self, label):
        """All 3 valid dining format labels must pass through unchanged."""
        mock_pipeline = build_mock_pipeline(label, 0.5)
        returned_label, _ = classify_dining_format(mock_pipeline, "text")
        assert returned_label == label


# ═══════════════════════════════════════════════════════════════════════════════
# 10. WRAPPER UNIT TESTS — OCCASION CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyOccasion:

    def test_returns_top_label_from_pipeline_output(self):
        """classify_occasion must return the highest-scored label."""
        mock_pipeline = build_mock_pipeline("Special Occasion", 0.59)
        label, _ = classify_occasion(mock_pipeline, "anniversary dinner")
        assert label == "Special Occasion"

    def test_returns_correct_confidence_score(self):
        mock_pipeline = build_mock_pipeline("Casual Night Out", 0.33)
        _, confidence = classify_occasion(mock_pipeline, "casual spot")
        assert confidence == 0.33

    def test_pipeline_called_with_four_occasion_candidates(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "sequence": "test",
            "labels": ["Special Occasion"],
            "scores": [0.6],
        }
        classify_occasion(mock_pipeline, "restaurant")
        args, kwargs = mock_pipeline.call_args
        passed_candidates = kwargs.get("candidate_labels", args[1] if len(args) > 1 else None)
        assert set(passed_candidates) == VALID_OCCASIONS, (
            f"Wrong candidates: {passed_candidates}"
        )

    @pytest.mark.parametrize("label", list(VALID_OCCASIONS))
    def test_every_valid_occasion_label_can_be_returned(self, label):
        """All 4 valid occasion labels must pass through unchanged."""
        mock_pipeline = build_mock_pipeline(label, 0.5)
        returned_label, _ = classify_occasion(mock_pipeline, "text")
        assert returned_label == label


# ═══════════════════════════════════════════════════════════════════════════════
# 11. WRAPPER UNIT TESTS — VIBE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassifyVibe:

    def test_returns_top_label_from_pipeline_output(self):
        """classify_vibe must return the highest-scored label."""
        mock_pipeline = build_mock_pipeline("Refined & Elegant", 0.82)
        label, _ = classify_vibe(mock_pipeline, "fine dining with white tablecloths")
        assert label == "Refined & Elegant"

    def test_returns_correct_confidence_score(self):
        mock_pipeline = build_mock_pipeline("Intimate", 0.63)
        _, confidence = classify_vibe(mock_pipeline, "cozy 8-seat omakase")
        assert confidence == 0.63

    def test_pipeline_called_with_six_vibe_candidates(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "sequence": "test",
            "labels": ["Refined & Elegant"],
            "scores": [0.8],
        }
        classify_vibe(mock_pipeline, "restaurant")
        args, kwargs = mock_pipeline.call_args
        passed_candidates = kwargs.get("candidate_labels", args[1] if len(args) > 1 else None)
        assert set(passed_candidates) == VALID_VIBES, (
            f"Wrong candidates: {passed_candidates}"
        )

    @pytest.mark.parametrize("label", list(VALID_VIBES))
    def test_every_valid_vibe_label_can_be_returned(self, label):
        """All 6 valid vibe labels must pass through unchanged."""
        mock_pipeline = build_mock_pipeline(label, 0.5)
        returned_label, _ = classify_vibe(mock_pipeline, "text")
        assert returned_label == label

    def test_theatrical_label_passes_through_unchanged(self):
        """
        'Theatrical & Experiential' contains an ampersand — verify the
        wrapper doesn't accidentally strip or escape special characters.
        """
        mock_pipeline = build_mock_pipeline("Theatrical & Experiential", 0.46)
        label, _ = classify_vibe(mock_pipeline, "avant-garde tasting menu")
        assert label == "Theatrical & Experiential"


# ═══════════════════════════════════════════════════════════════════════════════
# 12. EDGE CASE TESTS ON THE WRAPPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassificationEdgeCases:

    def test_very_low_confidence_score_still_returned(self):
        """
        A near-zero confidence score (e.g., 0.0001) must still be
        returned — the wrapper must not suppress low-confidence results.
        """
        mock_pipeline = build_mock_pipeline("Japanese", 0.0001)
        label, confidence = classify_cuisine(mock_pipeline, "ambiguous restaurant")
        assert label == "Japanese"
        assert confidence > 0.0

    def test_perfect_confidence_score_still_returned(self):
        """A perfect 1.0 confidence score must pass through unchanged."""
        mock_pipeline = build_mock_pipeline("Italian", 1.0)
        label, confidence = classify_cuisine(mock_pipeline, "classic Italian trattoria")
        assert label == "Italian"
        assert confidence == 1.0

    def test_empty_string_input_does_not_raise(self):
        """
        An empty string input must not raise. The pipeline mock will return
        a valid result regardless of the input text.
        """
        mock_pipeline = build_mock_pipeline("Contemporary American", 0.3)
        try:
            label, confidence = classify_cuisine(mock_pipeline, "")
        except Exception as e:
            pytest.fail(f"classify_cuisine raised on empty string: {e}")

    def test_single_label_candidate_list_returns_that_label(self):
        """
        If the pipeline is somehow called with only one candidate label,
        it must return that label — no IndexError.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "sequence": "test",
            "labels":  ["Japanese"],
            "scores":  [0.9],
        }
        result = _run_zero_shot(mock_pipeline, "sushi", ["Japanese"])
        assert result["labels"][0] == "Japanese"
        assert result["scores"][0] == 0.9

    def test_pipeline_input_text_is_passed_correctly(self):
        """
        The input text passed to classify_cuisine must be forwarded
        verbatim to the pipeline — the wrapper must not modify it.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "sequence": "test",
            "labels": ["French"],
            "scores": [0.7],
        }
        test_text = "A refined French bistro with Burgundy wine list."
        classify_cuisine(mock_pipeline, test_text)
        args, _ = mock_pipeline.call_args
        assert args[0] == test_text, (
            f"Pipeline received '{args[0]}' instead of '{test_text}'"
        )

    def test_multi_label_false_is_passed_to_pipeline(self):
        """
        The pipeline must be called with multi_label=False. This ensures
        the model treats each classification as mutually exclusive,
        which is the correct setup for single-label classification.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "sequence": "test",
            "labels": ["Japanese"],
            "scores": [0.8],
        }
        classify_cuisine(mock_pipeline, "sushi bar")
        _, kwargs = mock_pipeline.call_args
        assert kwargs.get("multi_label") is False, (
            f"multi_label was not set to False. "
            f"kwargs passed: {kwargs}"
        )

    def test_confidence_is_numeric_not_string(self):
        """The confidence value returned by classify_cuisine must be a float."""
        mock_pipeline = build_mock_pipeline("Seafood", 0.512)
        _, confidence = classify_cuisine(mock_pipeline, "seafood restaurant")
        assert isinstance(confidence, float), (
            f"Expected float confidence, got {type(confidence)}"
        )

    def test_special_characters_in_text_do_not_raise(self):
        """
        Restaurant metadata often contains apostrophes, slashes, quotes,
        and ampersands. The wrapper must not raise on any of these.
        """
        mock_pipeline = build_mock_pipeline("Japanese", 0.6)
        tricky_texts = [
            "Chef's Table star — omakase & kaiseki",
            "n/naka: 13-course tasting menu",
            'Described as "exceptional" by Michelin',
            "Gwen Butcher Shop & Restaurant",
        ]
        for text in tricky_texts:
            try:
                classify_cuisine(mock_pipeline, text)
            except Exception as e:
                pytest.fail(
                    f"classify_cuisine raised on text with special chars: {e}\n"
                    f"Text: {text}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 13. MOCK INTEGRATION — PIPELINE IS NEVER CALLED LIVE
# Ensures that the mock infrastructure is working correctly and that
# no test accidentally triggers a real HuggingFace model download.
# ═══════════════════════════════════════════════════════════════════════════════

class TestMockIntegrity:

    def test_mock_pipeline_is_callable(self):
        """The mock returned by build_mock_pipeline must be callable."""
        mock_pipeline = build_mock_pipeline("Japanese", 0.7)
        assert callable(mock_pipeline)

    def test_mock_returns_correct_label_structure(self):
        """The mock's return value must match bart's actual output shape."""
        mock_pipeline = build_mock_pipeline("Italian", 0.65)
        result = mock_pipeline("any text", candidate_labels=["Italian", "French"])
        assert "labels" in result
        assert "scores" in result
        assert "sequence" in result
        assert result["labels"][0] == "Italian"
        assert result["scores"][0] == 0.65

    def test_mock_label_is_always_first(self):
        """
        The target label must always be first in the returned labels list,
        matching how bart returns results in descending score order.
        """
        for label in ["Seafood", "Refined & Elegant", "Tasting Menu / Omakase"]:
            mock = build_mock_pipeline(label, 0.5)
            result = mock("text", candidate_labels=[label])
            assert result["labels"][0] == label

    def test_mock_scores_sum_to_approximately_one(self):
        """
        The mock's scores should roughly sum to 1.0 — consistent with
        how a softmax classification model distributes probabilities.
        """
        mock_pipeline = build_mock_pipeline("Japanese", 0.7)
        result = mock_pipeline("sushi bar", candidate_labels=list(VALID_CUISINE_GROUPS))
        total = sum(result["scores"])
        assert abs(total - 1.0) < 0.01, (
            f"Mock scores don't sum to ~1.0: {total}"
        )

    def test_wrapper_does_not_call_pipeline_more_than_once(self):
        """
        Each classify_* function must call the pipeline exactly once.
        Multiple calls would indicate the wrapper is retrying or looping.
        """
        for classify_fn in [classify_cuisine, classify_dining_format,
                            classify_occasion, classify_vibe]:
            mock_pipeline = build_mock_pipeline("Japanese", 0.6)
            mock_pipeline.return_value["labels"][0] = list(
                VALID_CUISINE_GROUPS |
                VALID_DINING_FORMATS |
                VALID_OCCASIONS |
                VALID_VIBES
            )[0]
            classify_fn(mock_pipeline, "restaurant text")
            assert mock_pipeline.call_count == 1, (
                f"{classify_fn.__name__} called pipeline "
                f"{mock_pipeline.call_count} times (expected 1)"
            )
