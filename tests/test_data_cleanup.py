"""
tests/test_data_cleanup.py
==========================
Unit tests for Notebook 1 — Data Cleanup & Metadata Enrichment.

Covers:
  - 15-column schema integrity (all required columns present, no extras)
  - Row count (always 71)
  - Whitespace stripping in Address and Description columns
  - restaurant_metadata string format and content rules
  - Michelin label mapping (raw CSV value → formatted metadata label)
  - Rooftop suffix logic (appended only when Sky-High Rooftop == "Yes")
  - Customer Rating formatting (e.g., 5.0 → "5.0/5")
  - Multi-location restaurant handling (Badmaash, Pine & Crane)
  - No nulls in critical columns
  - Price range passthrough ($ symbols preserved as-is)

These tests do NOT depend on any HuggingFace model or external API.
They operate purely on pandas DataFrames and the metadata generation
logic extracted from Notebook 1.
"""

import re

import pandas as pd
import pytest

# ── Path constants ────────────────────────────────────────────────────────────
FINAL_CSV = "data/cleaned_restaurants_final.csv"

# ── Expected schema ───────────────────────────────────────────────────────────
EXPECTED_COLUMNS = [
    "Name",
    "Location",
    "Description",
    "Address",
    "Telephone Number",
    "Price",
    "Cuisine Type",
    "Dining Atmosphere",
    "Sky-High Rooftop",
    "Michelin-Guide",
    "Customer Ratings",
    "Operation Hours",
    "Reservations",
    "Dress Code",
    "restaurant_metadata",
]

# ── Michelin label map (raw CSV value → expected string in metadata) ──────────
MICHELIN_LABEL_MAP = {
    "3-Star":           "Michelin 3-Star",
    "2-Star":           "Michelin 2-Star",
    "1-Star":           "Michelin 1-Star",
    "Bib-Gourmand":     "Michelin Bib-Gourmand",
    "Michelin-Selected": "Michelin-Selected",
    "No":               "No",
}

# ── Restaurants known to have multiple locations ──────────────────────────────
MULTI_LOCATION_NAMES = {"Badmaash", "Pine & Crane"}

# ── Restaurants that should have the rooftop suffix ──────────────────────────
EXPECTED_ROOFTOP_NAMES = {
    "71Above",
    "La Boucherie",
    "Aperture at City Club LA",
    "Elephante",
    "Yamashiro",
    "LouLou",
}


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def df():
    """Load the final cleaned CSV once for the entire test module."""
    return pd.read_csv(FINAL_CSV)


@pytest.fixture(scope="module")
def rooftop_rows(df):
    """Subset of rows where Sky-High Rooftop == 'Yes'."""
    return df[df["Sky-High Rooftop"] == "Yes"]


@pytest.fixture(scope="module")
def non_rooftop_rows(df):
    """Subset of rows where Sky-High Rooftop == 'No'."""
    return df[df["Sky-High Rooftop"] == "No"]


# ═════════════════════════════════════════════════════════════════════════════
# 1. SCHEMA TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestSchema:

    def test_column_count_is_15(self, df):
        """Final CSV must have exactly 15 columns — 14 base + restaurant_metadata."""
        assert len(df.columns) == 15, (
            f"Expected 15 columns, found {len(df.columns)}.\n"
            f"Columns present: {list(df.columns)}"
        )

    def test_all_required_columns_present(self, df):
        """Every column name in EXPECTED_COLUMNS must exist in the DataFrame."""
        missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_no_unexpected_columns(self, df):
        """No extra columns beyond the expected 15 should be present."""
        extra = [col for col in df.columns if col not in EXPECTED_COLUMNS]
        assert not extra, f"Unexpected extra columns found: {extra}"

    def test_column_order_matches_expected(self, df):
        """Column order must match EXPECTED_COLUMNS exactly."""
        assert list(df.columns) == EXPECTED_COLUMNS, (
            f"Column order mismatch.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Got:      {list(df.columns)}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 2. ROW COUNT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestRowCount:

    def test_total_row_count_is_71(self, df):
        """Dataset must contain exactly 71 restaurant records."""
        assert len(df) == 71, f"Expected 71 rows, got {len(df)}"

    def test_multi_location_badmaash_has_two_rows(self, df):
        """Badmaash intentionally has two locations — both rows must be present."""
        count = len(df[df["Name"] == "Badmaash"])
        assert count == 2, (
            f"Expected 2 rows for Badmaash (Hollywood + DTLA), found {count}"
        )

    def test_multi_location_pine_and_crane_has_two_rows(self, df):
        """Pine & Crane intentionally has two locations — both rows must be present."""
        count = len(df[df["Name"] == "Pine & Crane"])
        assert count == 2, (
            f"Expected 2 rows for Pine & Crane (Silver Lake + DTLA), found {count}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. NULL / MISSING VALUE TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestNullValues:

    @pytest.mark.parametrize("col", [
        "Name", "Location", "Description", "Address",
        "Price", "Cuisine Type", "Dining Atmosphere",
        "Sky-High Rooftop", "Michelin-Guide",
        "Customer Ratings", "restaurant_metadata",
    ])
    def test_no_nulls_in_critical_column(self, df, col):
        """Critical columns must have zero null values."""
        null_count = df[col].isnull().sum()
        assert null_count == 0, (
            f"Column '{col}' has {null_count} null value(s):\n"
            f"{df[df[col].isnull()][['Name', col]]}"
        )

    def test_customer_ratings_are_numeric(self, df):
        """Customer Ratings must be parseable as floats."""
        assert pd.to_numeric(df["Customer Ratings"], errors="coerce").notnull().all(), \
            "One or more Customer Ratings values are not numeric"

    def test_customer_ratings_in_valid_range(self, df):
        """All customer ratings must be between 1.0 and 5.0 inclusive."""
        ratings = df["Customer Ratings"].astype(float)
        out_of_range = df[(ratings < 1.0) | (ratings > 5.0)][["Name", "Customer Ratings"]]
        assert out_of_range.empty, (
            f"Customer ratings out of valid range (1.0-5.0):\n{out_of_range}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4. WHITESPACE TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestWhitespace:

    def test_no_leading_trailing_whitespace_in_description(self, df):
        """Description column must have no leading or trailing whitespace."""
        bad_rows = df[df["Description"] != df["Description"].str.strip()][["Name", "Description"]]
        assert bad_rows.empty, (
            f"Leading/trailing whitespace found in Description:\n{bad_rows}"
        )

    def test_no_leading_trailing_whitespace_in_address(self, df):
        """Address column must have no leading or trailing whitespace."""
        bad_rows = df[df["Address"] != df["Address"].str.strip()][["Name", "Address"]]
        assert bad_rows.empty, (
            f"Leading/trailing whitespace found in Address:\n{bad_rows}"
        )

    def test_no_leading_trailing_whitespace_in_name(self, df):
        """Name column must have no leading or trailing whitespace."""
        bad_rows = df[df["Name"] != df["Name"].str.strip()][["Name"]]
        assert bad_rows.empty, (
            f"Leading/trailing whitespace found in Name:\n{bad_rows}"
        )

    def test_no_leading_trailing_whitespace_in_metadata(self, df):
        """restaurant_metadata column must have no leading or trailing whitespace."""
        bad_rows = df[
            df["restaurant_metadata"] != df["restaurant_metadata"].str.strip()
        ][["Name", "restaurant_metadata"]]
        assert bad_rows.empty, (
            f"Leading/trailing whitespace in restaurant_metadata:\n{bad_rows['Name'].tolist()}"
        )

    def test_morihiro_address_has_correct_zip(self, df):
        """
        Morihiro's address was corrected during cleanup to use the Echo Park
        zip code (90026). Regression test to ensure it doesn't revert.
        """
        morihiro = df[df["Name"] == "Morihiro"]
        assert len(morihiro) == 1, "Expected exactly one Morihiro row"
        address = morihiro["Address"].iloc[0]
        assert "90026" in address, (
            f"Morihiro zip code regression — expected 90026 in address, got: '{address}'"
        )

    def test_mastros_address_has_comma(self, df):
        """
        Mastro's Ocean Club address was corrected to include a missing comma.
        Regression test to confirm the fix persists.
        """
        mastros = df[df["Name"].str.contains("Mastro", na=False)]
        assert len(mastros) >= 1, "Expected at least one Mastro's row"
        for _, row in mastros.iterrows():
            assert "," in row["Address"], (
                f"Mastro's Ocean Club address missing comma: '{row['Address']}'"
            )


# ═════════════════════════════════════════════════════════════════════════════
# 5. METADATA FORMAT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestMetadataFormat:
    """
    The restaurant_metadata field follows this template:
    "{Name} is a {Cuisine Type} restaurant located in {Location}, Los Angeles.
    {Description} Price range: {Price}. Atmosphere: {Dining Atmosphere}.
    Michelin Guide: {mapped_michelin_label}. Customer Rating: {Rating}/5."

    Rooftop restaurants append: " Rooftop/top-floor dining."
    """

    def test_metadata_ends_with_period_or_rooftop_suffix(self, df):
        """Every metadata string must end with '.' or 'Rooftop/top-floor dining.'"""
        valid_endings = (".", "Rooftop/top-floor dining.")
        bad = df[~df["restaurant_metadata"].str.endswith(valid_endings)][
            ["Name", "restaurant_metadata"]
        ]
        assert bad.empty, (
            f"Metadata strings with invalid endings:\n{bad['Name'].tolist()}"
        )

    def test_metadata_contains_restaurant_name(self, df):
        """Each metadata string must begin with the restaurant's own name."""
        bad_rows = []
        for _, row in df.iterrows():
            if not row["restaurant_metadata"].startswith(row["Name"]):
                bad_rows.append(row["Name"])
        assert not bad_rows, (
            f"Metadata does not start with the restaurant name:\n{bad_rows}"
        )

    def test_metadata_contains_los_angeles(self, df):
        """Every metadata string must include 'Los Angeles' as the county anchor."""
        bad = df[~df["restaurant_metadata"].str.contains("Los Angeles", na=False)][
            ["Name", "restaurant_metadata"]
        ]
        assert bad.empty, (
            f"'Los Angeles' missing from metadata:\n{bad['Name'].tolist()}"
        )

    def test_metadata_contains_price_range_label(self, df):
        """Every metadata string must contain 'Price range:' followed by $ symbols."""
        pattern = r"Price range: \$+"
        bad = df[~df["restaurant_metadata"].str.contains(pattern, regex=True, na=False)][
            ["Name", "restaurant_metadata"]
        ]
        assert bad.empty, (
            f"'Price range: $...' pattern missing from metadata:\n{bad['Name'].tolist()}"
        )

    def test_metadata_contains_atmosphere_label(self, df):
        """Every metadata string must contain 'Atmosphere:' followed by a value."""
        bad = df[~df["restaurant_metadata"].str.contains("Atmosphere:", na=False)][
            ["Name", "restaurant_metadata"]
        ]
        assert bad.empty, (
            f"'Atmosphere:' missing from metadata:\n{bad['Name'].tolist()}"
        )

    def test_metadata_contains_michelin_guide_label(self, df):
        """Every metadata string must contain 'Michelin Guide:' followed by a value."""
        bad = df[~df["restaurant_metadata"].str.contains("Michelin Guide:", na=False)][
            ["Name", "restaurant_metadata"]
        ]
        assert bad.empty, (
            f"'Michelin Guide:' missing from metadata:\n{bad['Name'].tolist()}"
        )

    def test_metadata_contains_customer_rating_label(self, df):
        """Every metadata string must contain 'Customer Rating:' with /5 format."""
        pattern = r"Customer Rating: \d+\.\d/5\."
        bad = df[~df["restaurant_metadata"].str.contains(pattern, regex=True, na=False)][
            ["Name", "restaurant_metadata"]
        ]
        assert bad.empty, (
            f"'Customer Rating: X.X/5.' pattern missing or malformed:\n{bad['Name'].tolist()}"
        )

    def test_metadata_price_matches_csv_price(self, df):
        """The price symbols in metadata must exactly match the Price column."""
        for _, row in df.iterrows():
            price = row["Price"]
            metadata = row["restaurant_metadata"]
            expected_snippet = f"Price range: {price}."
            assert expected_snippet in metadata, (
                f"{row['Name']}: expected '{expected_snippet}' in metadata.\n"
                f"Metadata: {metadata}"
            )

    def test_metadata_atmosphere_matches_csv_atmosphere(self, df):
        """The atmosphere value in metadata must match the Dining Atmosphere column."""
        for _, row in df.iterrows():
            atmosphere = row["Dining Atmosphere"]
            metadata = row["restaurant_metadata"]
            expected_snippet = f"Atmosphere: {atmosphere}."
            assert expected_snippet in metadata, (
                f"{row['Name']}: expected '{expected_snippet}' in metadata.\n"
                f"Metadata: {metadata}"
            )

    def test_metadata_rating_matches_csv_rating(self, df):
        """The customer rating in metadata must match the Customer Ratings column."""
        for _, row in df.iterrows():
            rating = row["Customer Ratings"]
            metadata = row["restaurant_metadata"]
            expected_snippet = f"Customer Rating: {rating}/5."
            assert expected_snippet in metadata, (
                f"{row['Name']}: expected '{expected_snippet}' in metadata.\n"
                f"Metadata: {metadata}"
            )

    def test_metadata_cuisine_type_matches_csv(self, df):
        """The cuisine type in metadata must match the Cuisine Type column."""
        for _, row in df.iterrows():
            cuisine = row["Cuisine Type"]
            metadata = row["restaurant_metadata"]
            # Pattern: "{Name} is a {Cuisine Type} restaurant"
            expected_snippet = f"is a {cuisine} restaurant"
            assert expected_snippet in metadata, (
                f"{row['Name']}: expected 'is a {cuisine} restaurant' in metadata.\n"
                f"Metadata: {metadata}"
            )

    def test_metadata_location_matches_csv(self, df):
        """The location in metadata must match the Location column."""
        for _, row in df.iterrows():
            location = row["Location"]
            metadata = row["restaurant_metadata"]
            expected_snippet = f"located in {location}, Los Angeles"
            assert expected_snippet in metadata, (
                f"{row['Name']}: expected 'located in {location}, Los Angeles' in metadata.\n"
                f"Metadata: {metadata}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# 6. MICHELIN LABEL MAPPING TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestMichelinLabelMapping:
    """
    Tests that raw Michelin-Guide CSV values are correctly transformed
    into their formatted equivalents in restaurant_metadata.
    """

    @pytest.mark.parametrize("raw_value,expected_label", MICHELIN_LABEL_MAP.items())
    def test_michelin_label_correctly_mapped(self, df, raw_value, expected_label):
        """
        For every restaurant with a given raw Michelin-Guide value,
        the metadata must contain the corresponding formatted label.
        """
        subset = df[df["Michelin-Guide"] == raw_value]
        if subset.empty:
            pytest.skip(f"No restaurants with Michelin-Guide == '{raw_value}' in dataset")
        for _, row in subset.iterrows():
            expected_snippet = f"Michelin Guide: {expected_label}."
            assert expected_snippet in row["restaurant_metadata"], (
                f"{row['Name']} (Michelin-Guide='{raw_value}'): "
                f"expected '{expected_snippet}' in metadata.\n"
                f"Metadata: {row['restaurant_metadata']}"
            )

    def test_all_michelin_values_are_known(self, df):
        """
        Every value in the Michelin-Guide column must be one of the
        recognized values in MICHELIN_LABEL_MAP. Unknown values would
        silently produce malformed metadata.
        """
        known = set(MICHELIN_LABEL_MAP.keys())
        actual = set(df["Michelin-Guide"].unique())
        unknown = actual - known
        assert not unknown, (
            f"Unrecognized Michelin-Guide values found: {unknown}\n"
            f"Add them to MICHELIN_LABEL_MAP and update the metadata generation logic."
        )

    def test_three_star_restaurants_present(self, df):
        """Dataset must include at least the two known 3-Star restaurants."""
        three_star = df[df["Michelin-Guide"] == "3-Star"]["Name"].tolist()
        assert len(three_star) >= 2, (
            f"Expected at least 2 Michelin 3-Star restaurants, found {len(three_star)}: {three_star}"
        )
        assert "Somni" in three_star, "Somni should be a 3-Star restaurant"
        assert "Providence" in three_star, "Providence should be a 3-Star restaurant"


# ═════════════════════════════════════════════════════════════════════════════
# 7. ROOFTOP SUFFIX TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestRooftopSuffix:
    """
    Restaurants with Sky-High Rooftop == 'Yes' must have
    ' Rooftop/top-floor dining.' appended to their metadata.
    Restaurants with Sky-High Rooftop == 'No' must NOT have this suffix.
    """

    ROOFTOP_SUFFIX = "Rooftop/top-floor dining."

    def test_rooftop_restaurants_have_suffix(self, rooftop_rows):
        """All Sky-High Rooftop == 'Yes' rows must end with the rooftop suffix."""
        missing_suffix = rooftop_rows[
            ~rooftop_rows["restaurant_metadata"].str.endswith(self.ROOFTOP_SUFFIX)
        ][["Name", "Sky-High Rooftop", "restaurant_metadata"]]
        assert missing_suffix.empty, (
            f"Rooftop restaurants missing ' Rooftop/top-floor dining.' suffix:\n"
            f"{missing_suffix['Name'].tolist()}"
        )

    def test_non_rooftop_restaurants_do_not_have_suffix(self, non_rooftop_rows):
        """All Sky-High Rooftop == 'No' rows must NOT contain the rooftop suffix."""
        has_suffix = non_rooftop_rows[
            non_rooftop_rows["restaurant_metadata"].str.contains(
                self.ROOFTOP_SUFFIX, na=False
            )
        ][["Name", "Sky-High Rooftop", "restaurant_metadata"]]
        assert has_suffix.empty, (
            f"Non-rooftop restaurants incorrectly have rooftop suffix:\n"
            f"{has_suffix['Name'].tolist()}"
        )

    def test_known_rooftop_names_are_flagged_yes(self, df):
        """
        Spot-check: the six known rooftop/top-floor restaurants must all
        have Sky-High Rooftop == 'Yes'.
        """
        for name in EXPECTED_ROOFTOP_NAMES:
            rows = df[df["Name"] == name]
            if rows.empty:
                pytest.skip(f"'{name}' not found in dataset")
            for _, row in rows.iterrows():
                assert row["Sky-High Rooftop"] == "Yes", (
                    f"'{name}' should be Sky-High Rooftop = 'Yes', got '{row['Sky-High Rooftop']}'"
                )

    def test_rooftop_column_only_contains_yes_or_no(self, df):
        """Sky-High Rooftop column must only ever contain 'Yes' or 'No'."""
        valid = {"Yes", "No"}
        actual = set(df["Sky-High Rooftop"].unique())
        invalid = actual - valid
        assert not invalid, (
            f"Unexpected values in Sky-High Rooftop column: {invalid}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 8. METADATA GENERATION LOGIC UNIT TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestMetadataGenerationLogic:
    """
    Tests the metadata generation function in isolation using
    synthetic row data — no CSV dependency. This verifies the
    construction logic independently of the actual dataset.
    """

    def _build_metadata(self, row: dict) -> str:
        """
        Mirrors the metadata generation logic from Notebook 1.
        Keep this in sync with the actual notebook implementation.
        """
        michelin_map = {
            "3-Star":            "Michelin 3-Star",
            "2-Star":            "Michelin 2-Star",
            "1-Star":            "Michelin 1-Star",
            "Bib-Gourmand":      "Michelin Bib-Gourmand",
            "Michelin-Selected": "Michelin-Selected",
            "No":                "No",
        }
        michelin_label = michelin_map.get(row["Michelin-Guide"], row["Michelin-Guide"])
        rating = float(row["Customer Ratings"])

        metadata = (
            f"{row['Name']} is a {row['Cuisine Type']} restaurant located in "
            f"{row['Location']}, Los Angeles. "
            f"{row['Description']} "
            f"Price range: {row['Price']}. "
            f"Atmosphere: {row['Dining Atmosphere']}. "
            f"Michelin Guide: {michelin_label}. "
            f"Customer Rating: {rating}/5."
        )

        if row.get("Sky-High Rooftop") == "Yes":
            metadata += " Rooftop/top-floor dining."

        return metadata

    def test_standard_non_rooftop_restaurant(self):
        """Basic metadata generation for a standard non-rooftop restaurant."""
        row = {
            "Name": "Test Bistro",
            "Cuisine Type": "French",
            "Location": "Beverly Hills",
            "Description": "An elegant French bistro.",
            "Price": "$$$$",
            "Dining Atmosphere": "Fine-Dining",
            "Michelin-Guide": "1-Star",
            "Customer Ratings": 4.5,
            "Sky-High Rooftop": "No",
        }
        result = self._build_metadata(row)
        assert result == (
            "Test Bistro is a French restaurant located in Beverly Hills, Los Angeles. "
            "An elegant French bistro. "
            "Price range: $$$$. "
            "Atmosphere: Fine-Dining. "
            "Michelin Guide: Michelin 1-Star. "
            "Customer Rating: 4.5/5."
        )

    def test_rooftop_restaurant_appends_suffix(self):
        """Rooftop flag must append the suffix after the final period."""
        row = {
            "Name": "Sky Lounge",
            "Cuisine Type": "Contemporary American",
            "Location": "Downtown Los Angeles",
            "Description": "A sky-high dining experience.",
            "Price": "$$$$",
            "Dining Atmosphere": "Fine-Dining",
            "Michelin-Guide": "No",
            "Customer Ratings": 4.7,
            "Sky-High Rooftop": "Yes",
        }
        result = self._build_metadata(row)
        assert result.endswith("Rooftop/top-floor dining.")
        assert "Customer Rating: 4.7/5." in result

    def test_bib_gourmand_label_mapping(self):
        """'Bib-Gourmand' raw value must map to 'Michelin Bib-Gourmand' in output."""
        row = {
            "Name": "Value Place",
            "Cuisine Type": "Italian",
            "Location": "Hollywood",
            "Description": "Great value Italian.",
            "Price": "$$",
            "Dining Atmosphere": "Casual",
            "Michelin-Guide": "Bib-Gourmand",
            "Customer Ratings": 4.6,
            "Sky-High Rooftop": "No",
        }
        result = self._build_metadata(row)
        assert "Michelin Guide: Michelin Bib-Gourmand." in result

    def test_no_michelin_label_mapping(self):
        """'No' raw Michelin value must render as 'No' (not blank or 'None')."""
        row = {
            "Name": "Regular Place",
            "Cuisine Type": "Steakhouse",
            "Location": "Santa Monica",
            "Description": "A solid steakhouse.",
            "Price": "$$$$",
            "Dining Atmosphere": "Fine-Dining",
            "Michelin-Guide": "No",
            "Customer Ratings": 4.3,
            "Sky-High Rooftop": "No",
        }
        result = self._build_metadata(row)
        assert "Michelin Guide: No." in result
        assert "Michelin Guide: None" not in result

    def test_five_dollar_sign_price_preserved(self):
        """Ultra-luxury price ($$$$$) must pass through without truncation."""
        row = {
            "Name": "Somni",
            "Cuisine Type": "Spanish Modernist",
            "Location": "West Hollywood",
            "Description": "A 14-seat Spanish Modernist chef's counter.",
            "Price": "$$$$$",
            "Dining Atmosphere": "Fine-Dining",
            "Michelin-Guide": "3-Star",
            "Customer Ratings": 5.0,
            "Sky-High Rooftop": "No",
        }
        result = self._build_metadata(row)
        assert "Price range: $$$$$." in result

    def test_integer_like_rating_renders_with_decimal(self):
        """Rating of 5.0 must render as '5.0/5' not '5/5'."""
        row = {
            "Name": "Perfect Place",
            "Cuisine Type": "Japanese",
            "Location": "Little Tokyo",
            "Description": "Flawless omakase.",
            "Price": "$$$$",
            "Dining Atmosphere": "Fine-Dining",
            "Michelin-Guide": "1-Star",
            "Customer Ratings": 5.0,
            "Sky-High Rooftop": "No",
        }
        result = self._build_metadata(row)
        assert "Customer Rating: 5.0/5." in result
        assert "Customer Rating: 5/5." not in result

    def test_special_characters_in_name_preserved(self):
        """
        Restaurant names with special characters (ampersands, slashes,
        apostrophes) must be preserved exactly in metadata output.
        """
        row = {
            "Name": "Gwen Butcher Shop & Restaurant",
            "Cuisine Type": "American Steakhouse",
            "Location": "Hollywood",
            "Description": "A glamorous steakhouse.",
            "Price": "$$$$",
            "Dining Atmosphere": "Fine-Dining",
            "Michelin-Guide": "1-Star",
            "Customer Ratings": 3.0,
            "Sky-High Rooftop": "No",
        }
        result = self._build_metadata(row)
        assert result.startswith("Gwen Butcher Shop & Restaurant is a")

    def test_slash_in_cuisine_type_preserved(self):
        """Cuisine types with slashes (e.g. 'French / Contemporary American') must be preserved."""
        row = {
            "Name": "Melisse",
            "Cuisine Type": "French / Contemporary American",
            "Location": "Santa Monica",
            "Description": "A hidden fine dining alcove.",
            "Price": "$$$$",
            "Dining Atmosphere": "Fine-Dining",
            "Michelin-Guide": "2-Star",
            "Customer Ratings": 4.0,
            "Sky-High Rooftop": "No",
        }
        result = self._build_metadata(row)
        assert "is a French / Contemporary American restaurant" in result
