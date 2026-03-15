"""
tests/test_vector_search.py
===========================
Unit tests for Notebook 2 — Vector Search with ChromaDB.

Strategy
--------
All ChromaDB and SentenceTransformer calls are fully mocked.
No GPU, no model downloads, no persisted chroma_restaurants/ directory
required. Tests operate against a deterministic in-memory fake corpus
of 71 restaurant records derived directly from the real CSV.

The fake ChromaDB collection mirrors the exact document + metadata
structure used in the real notebook:
  - document : restaurant_metadata string
  - metadata : {Name, Michelin-Guide, Price, Dining Atmosphere,
                Sky-High Rooftop, Location}

The retrieval function under test has this signature (from Notebook 2):
  retrieve_restaurants(
      query,
      top_k=5,
      michelin_filter=None,
      price_filter=None,
      atmosphere_filter=None,
      rooftop_only=False,
  ) -> list[dict]

Covers
------
  1. Basic retrieval — returns correct number of results, correct shape
  2. michelin_filter — exact value matching, multi-value list, None passthrough
  3. price_filter — exact value matching, None passthrough
  4. atmosphere_filter — exact value matching, None passthrough
  5. rooftop_only flag — returns only Sky-High Rooftop == "Yes" rows
  6. Combined filters — multiple active filters applied together
  7. top_k boundary — returns exactly top_k, or all available if fewer
  8. Candidate pool strategy — never returns fewer than top_k when pool
     is large enough; degrades gracefully when dataset is truly smaller
  9. Filter isolation — active filters never bleed non-matching rows into results
 10. Edge cases — top_k=1, top_k larger than corpus, empty query string,
     filter with no matches

Real data counts (from cleaned_restaurants_final.csv):
  Total restaurants:       71
  Michelin 1-Star:         20
  Michelin 2-Star:          4
  Michelin 3-Star:          2
  Michelin Bib-Gourmand:   10
  Michelin-Selected:       11
  No Michelin:             24
  Price $$$$:              42
  Price $$$$$:              5
  Price $$$:               12
  Price $$:                 9
  Price $:                  3
  Fine-Dining atmosphere:  40
  Casual atmosphere:       14
  Rooftop (Yes):            6
"""

from __future__ import annotations

import random
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Path constants ─────────────────────────────────────────────────────────────
FINAL_CSV = "data/cleaned_restaurants_final.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# FAKE CHROMADB INFRASTRUCTURE
#
# Builds a deterministic in-memory corpus from the real CSV so that every
# assertion is grounded in actual data counts, not invented magic numbers.
# ═══════════════════════════════════════════════════════════════════════════════

def _load_corpus() -> list[dict]:
    """
    Load the real CSV and build the list of ChromaDB document dicts that
    Notebook 2 would have ingested.  Each dict has:
        id       - restaurant Name (unique per row, collision-safe)
        document - restaurant_metadata string
        metadata - the filterable fields stored as ChromaDB metadata
    """
    df = pd.read_csv(FINAL_CSV)
    corpus = []
    for i, row in df.iterrows():
        corpus.append({
            "id": f"{row['Name']}_{i}",   # suffix avoids duplicate-name collision
            "document": row["restaurant_metadata"],
            "metadata": {
                "Name":             row["Name"],
                "Michelin-Guide":   row["Michelin-Guide"],
                "Price":            row["Price"],
                "Dining Atmosphere": row["Dining Atmosphere"],
                "Sky-High Rooftop": row["Sky-High Rooftop"],
                "Location":         row["Location"],
            },
        })
    return corpus


# Load once at module level — shared across all tests
_CORPUS: list[dict] = _load_corpus()


def _apply_filters(
    corpus: list[dict],
    michelin_filter: str | list[str] | None,
    price_filter: str | list[str] | None,
    atmosphere_filter: str | None,
    rooftop_only: bool,
) -> list[dict]:
    """
    Pure-Python replica of the ChromaDB where-clause filter logic.

    ChromaDB $eq / $in behaviour:
      - Single string  → exact match  ($eq)
      - List of strings → any-of match ($in)
      - None           → no filter applied
    """
    results = corpus[:]

    if michelin_filter is not None:
        if isinstance(michelin_filter, list):
            results = [r for r in results
                       if r["metadata"]["Michelin-Guide"] in michelin_filter]
        else:
            results = [r for r in results
                       if r["metadata"]["Michelin-Guide"] == michelin_filter]

    if price_filter is not None:
        if isinstance(price_filter, list):
            results = [r for r in results
                       if r["metadata"]["Price"] in price_filter]
        else:
            results = [r for r in results
                       if r["metadata"]["Price"] == price_filter]

    if atmosphere_filter is not None:
        results = [r for r in results
                   if r["metadata"]["Dining Atmosphere"] == atmosphere_filter]

    if rooftop_only:
        results = [r for r in results
                   if r["metadata"]["Sky-High Rooftop"] == "Yes"]

    return results


class FakeCollection:
    """
    Minimal ChromaDB Collection stand-in.

    .query() returns ChromaDB's actual response shape:
        {
          "ids":       [[id, ...]],
          "documents": [[doc, ...]],
          "metadatas": [[meta, ...]],
          "distances": [[float, ...]],
        }

    The candidate pool strategy means the real notebook queries with
    n_results=candidate_pool_size first, then filters in Python.
    We replicate that by returning all corpus docs ranked by a
    deterministic fake similarity score (index position), then the
    retrieval function trims to top_k after applying filters.
    """

    def __init__(self, corpus: list[dict]):
        self._corpus = corpus

    def query(
        self,
        query_texts: list[str],
        n_results: int,
        where: dict | None = None,
    ) -> dict:
        # Respect n_results cap but ignore embedding similarity —
        # return first n_results docs in corpus order (deterministic).
        pool = self._corpus[:n_results]
        return {
            "ids":       [[d["id"] for d in pool]],
            "documents": [[d["document"] for d in pool]],
            "metadatas": [[d["metadata"] for d in pool]],
            "distances": [[float(i) * 0.1 for i in range(len(pool))]],
        }

    def count(self) -> int:
        return len(self._corpus)


# ═══════════════════════════════════════════════════════════════════════════════
# THE RETRIEVAL FUNCTION (mirrors Notebook 2 implementation)
#
# This is the function under test.  It is defined here so tests can run
# without importing the actual notebook.  Keep it in sync with the real
# implementation in 2_Restaurant_Vector_Search.ipynb.
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_restaurants(
    query: str,
    collection: Any,
    top_k: int = 5,
    michelin_filter: str | list[str] | None = None,
    price_filter: str | list[str] | None = None,
    atmosphere_filter: str | None = None,
    rooftop_only: bool = False,
) -> list[dict]:
    """
    Semantic restaurant retrieval with metadata filtering.

    Candidate pool strategy:
      Query ChromaDB for a large pool (min 20, or top_k * 4),
      then apply metadata filters in Python to prevent filter depletion.
    """
    candidate_pool_size = max(20, top_k * 4)

    raw = collection.query(
        query_texts=[query],
        n_results=min(candidate_pool_size, collection.count()),
    )

    ids        = raw["ids"][0]
    documents  = raw["documents"][0]
    metadatas  = raw["metadatas"][0]
    distances  = raw["distances"][0]

    results = []
    for id_, doc, meta, dist in zip(ids, documents, metadatas, distances):
        # Apply filters in Python after retrieval
        if michelin_filter is not None:
            if isinstance(michelin_filter, list):
                if meta["Michelin-Guide"] not in michelin_filter:
                    continue
            else:
                if meta["Michelin-Guide"] != michelin_filter:
                    continue

        if price_filter is not None:
            if isinstance(price_filter, list):
                if meta["Price"] not in price_filter:
                    continue
            else:
                if meta["Price"] != price_filter:
                    continue

        if atmosphere_filter is not None:
            if meta["Dining Atmosphere"] != atmosphere_filter:
                continue

        if rooftop_only and meta["Sky-High Rooftop"] != "Yes":
            continue

        results.append({
            "id":       id_,
            "document": doc,
            "metadata": meta,
            "distance": dist,
        })

        if len(results) == top_k:
            break

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def df():
    """Real CSV DataFrame — used for ground-truth count assertions."""
    return pd.read_csv(FINAL_CSV)


@pytest.fixture(scope="module")
def collection():
    """Fake ChromaDB collection backed by the full 71-restaurant corpus."""
    return FakeCollection(_CORPUS)


@pytest.fixture(scope="module")
def df_counts(df):
    """Pre-computed counts for every filter value — avoids recalculating per test."""
    return {
        "total":               len(df),
        "michelin_1star":      len(df[df["Michelin-Guide"] == "1-Star"]),
        "michelin_2star":      len(df[df["Michelin-Guide"] == "2-Star"]),
        "michelin_3star":      len(df[df["Michelin-Guide"] == "3-Star"]),
        "michelin_bib":        len(df[df["Michelin-Guide"] == "Bib-Gourmand"]),
        "michelin_selected":   len(df[df["Michelin-Guide"] == "Michelin-Selected"]),
        "michelin_no":         len(df[df["Michelin-Guide"] == "No"]),
        "price_4dollar":       len(df[df["Price"] == "$$$$"]),
        "price_5dollar":       len(df[df["Price"] == "$$$$$"]),
        "price_3dollar":       len(df[df["Price"] == "$$$"]),
        "price_2dollar":       len(df[df["Price"] == "$$"]),
        "price_1dollar":       len(df[df["Price"] == "$"]),
        "atmosphere_fine":     len(df[df["Dining Atmosphere"] == "Fine-Dining"]),
        "atmosphere_casual":   len(df[df["Dining Atmosphere"] == "Casual"]),
        "atmosphere_romantic": len(df[df["Dining Atmosphere"] == "Romantic"]),
        "rooftop_yes":         len(df[df["Sky-High Rooftop"] == "Yes"]),
        "rooftop_no":          len(df[df["Sky-High Rooftop"] == "No"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASIC RETRIEVAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBasicRetrieval:

    def test_returns_list(self, collection):
        """retrieve_restaurants must return a list."""
        results = retrieve_restaurants("romantic sushi dinner", collection)
        assert isinstance(results, list)

    def test_default_top_k_returns_five(self, collection):
        """Default top_k=5 must return exactly 5 results."""
        results = retrieve_restaurants("fine dining Japanese", collection)
        assert len(results) == 5

    def test_each_result_has_required_keys(self, collection):
        """Every result dict must have id, document, metadata, and distance keys."""
        results = retrieve_restaurants("steakhouse Beverly Hills", collection)
        for r in results:
            assert "id"       in r, f"Missing 'id' key in result: {r}"
            assert "document" in r, f"Missing 'document' key in result: {r}"
            assert "metadata" in r, f"Missing 'metadata' key in result: {r}"
            assert "distance" in r, f"Missing 'distance' key in result: {r}"

    def test_metadata_contains_required_fields(self, collection):
        """Each result's metadata must contain the six expected fields."""
        required_fields = {
            "Name", "Michelin-Guide", "Price",
            "Dining Atmosphere", "Sky-High Rooftop", "Location"
        }
        results = retrieve_restaurants("Italian pasta", collection)
        for r in results:
            missing = required_fields - set(r["metadata"].keys())
            assert not missing, (
                f"Result '{r['metadata'].get('Name', '?')}' missing metadata fields: {missing}"
            )

    def test_document_is_non_empty_string(self, collection):
        """Every result's document must be a non-empty string."""
        results = retrieve_restaurants("Michelin tasting menu", collection)
        for r in results:
            assert isinstance(r["document"], str), "document must be a string"
            assert len(r["document"]) > 0, "document must not be empty"

    def test_distance_is_float(self, collection):
        """Every result's distance must be a float."""
        results = retrieve_restaurants("rooftop views", collection)
        for r in results:
            assert isinstance(r["distance"], float), (
                f"distance must be float, got {type(r['distance'])}"
            )

    def test_empty_query_string_still_returns_results(self, collection):
        """An empty query string must not raise an exception."""
        results = retrieve_restaurants("", collection, top_k=5)
        assert isinstance(results, list)

    def test_results_do_not_contain_duplicates(self, collection):
        """No restaurant should appear twice in the results for a single query."""
        results = retrieve_restaurants("upscale dining", collection, top_k=10)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids)), (
            f"Duplicate restaurant IDs found in results: "
            f"{[x for x in ids if ids.count(x) > 1]}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MICHELIN FILTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMichelinFilter:

    @pytest.mark.parametrize("michelin_value", [
        "1-Star", "2-Star", "3-Star", "Bib-Gourmand", "Michelin-Selected", "No"
    ])
    def test_single_michelin_filter_excludes_others(self, collection, michelin_value):
        """
        When michelin_filter is set to a single value, every result must
        have exactly that Michelin-Guide value — no other tier may appear.
        """
        results = retrieve_restaurants(
            "best restaurant",
            collection,
            top_k=5,
            michelin_filter=michelin_value,
        )
        for r in results:
            assert r["metadata"]["Michelin-Guide"] == michelin_value, (
                f"Expected Michelin-Guide='{michelin_value}', "
                f"got '{r['metadata']['Michelin-Guide']}' for '{r['metadata']['Name']}'"
            )

    def test_michelin_filter_none_returns_all_tiers(self, collection):
        """
        michelin_filter=None must not restrict results by Michelin tier.
        All Michelin tiers should be eligible to appear.
        """
        results = retrieve_restaurants(
            "any restaurant",
            collection,
            top_k=15,
            michelin_filter=None,
        )
        tiers_found = {r["metadata"]["Michelin-Guide"] for r in results}
        # With 15 results from a diverse corpus, expect more than one tier
        assert len(tiers_found) > 1, (
            f"Expected multiple Michelin tiers with no filter, got: {tiers_found}"
        )

    def test_michelin_filter_list_accepts_multiple_tiers(self, collection):
        """
        michelin_filter as a list must accept restaurants from any listed tier
        and exclude all others.
        """
        allowed = ["1-Star", "2-Star"]
        results = retrieve_restaurants(
            "Japanese omakase",
            collection,
            top_k=10,
            michelin_filter=allowed,
        )
        for r in results:
            assert r["metadata"]["Michelin-Guide"] in allowed, (
                f"'{r['metadata']['Name']}' has tier '{r['metadata']['Michelin-Guide']}' "
                f"which is not in allowed list {allowed}"
            )

    def test_michelin_3star_filter_returns_only_somni_and_providence(
        self, collection, df_counts
    ):
        """
        Only 2 restaurants in the dataset are 3-Star (Somni and Providence).
        With top_k=5, we expect at most 2 results — the pool won't be depleted.
        """
        results = retrieve_restaurants(
            "exceptional dining",
            collection,
            top_k=5,
            michelin_filter="3-Star",
        )
        # Can't get more results than actually exist
        assert len(results) <= df_counts["michelin_3star"], (
            f"Got {len(results)} 3-Star results but only {df_counts['michelin_3star']} exist"
        )
        names = {r["metadata"]["Name"] for r in results}
        assert names.issubset({"Somni", "Providence"}), (
            f"3-Star filter returned unexpected restaurants: {names - {'Somni', 'Providence'}}"
        )

    def test_michelin_starred_only_list_covers_all_star_tiers(self, collection):
        """
        Filtering for ['1-Star','2-Star','3-Star'] must never return
        Bib-Gourmand, Michelin-Selected, or No restaurants.
        """
        starred_tiers = ["1-Star", "2-Star", "3-Star"]
        results = retrieve_restaurants(
            "Michelin starred restaurants",
            collection,
            top_k=10,
            michelin_filter=starred_tiers,
        )
        for r in results:
            assert r["metadata"]["Michelin-Guide"] in starred_tiers, (
                f"Non-starred restaurant slipped through: "
                f"'{r['metadata']['Name']}' ({r['metadata']['Michelin-Guide']})"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PRICE FILTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPriceFilter:

    @pytest.mark.parametrize("price_value", ["$", "$$", "$$$", "$$$$", "$$$$$"])
    def test_single_price_filter_excludes_other_tiers(self, collection, price_value):
        """
        When price_filter is set, every result must carry exactly that
        price symbol string — no other tier may appear.
        """
        results = retrieve_restaurants(
            "dinner tonight",
            collection,
            top_k=5,
            price_filter=price_value,
        )
        for r in results:
            assert r["metadata"]["Price"] == price_value, (
                f"Expected Price='{price_value}', "
                f"got '{r['metadata']['Price']}' for '{r['metadata']['Name']}'"
            )

    def test_price_filter_none_does_not_restrict(self, collection):
        """
        price_filter=None must allow all price tiers to appear.
        """
        results = retrieve_restaurants(
            "restaurant",
            collection,
            top_k=20,
            price_filter=None,
        )
        prices_found = {r["metadata"]["Price"] for r in results}
        assert len(prices_found) > 1, (
            f"Expected multiple price tiers with no filter, got: {prices_found}"
        )

    def test_price_filter_list_accepts_multiple_tiers(self, collection):
        """price_filter as a list must accept any of the listed price tiers."""
        allowed = ["$$$", "$$$$"]
        results = retrieve_restaurants(
            "fine dining",
            collection,
            top_k=10,
            price_filter=allowed,
        )
        for r in results:
            assert r["metadata"]["Price"] in allowed, (
                f"'{r['metadata']['Name']}' has price '{r['metadata']['Price']}' "
                f"not in allowed {allowed}"
            )

    def test_price_5dollar_filter_result_count_capped_by_dataset(
        self, collection, df_counts
    ):
        """
        Only 5 restaurants are priced at $$$$$. Requesting top_k=10
        must return at most 5, never more than what exists.
        """
        results = retrieve_restaurants(
            "ultra luxury dining",
            collection,
            top_k=10,
            price_filter="$$$$$",
        )
        assert len(results) <= df_counts["price_5dollar"], (
            f"Got {len(results)} $$$$$ results but only {df_counts['price_5dollar']} exist"
        )

    def test_price_dollar_sign_symbols_not_truncated(self, collection):
        """
        The filter must match on the full dollar-sign string.
        '$$$' must not accidentally match '$$$$' or '$$$$$'.
        """
        results = retrieve_restaurants(
            "affordable luxury",
            collection,
            top_k=5,
            price_filter="$$$",
        )
        for r in results:
            # Must be exactly $$$ — not $$ or $$$$ or $$$$$
            assert r["metadata"]["Price"] == "$$$", (
                f"Price filter '$$$' matched '{r['metadata']['Price']}' "
                f"for '{r['metadata']['Name']}' — dollar signs may be leaking"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ATMOSPHERE FILTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAtmosphereFilter:

    @pytest.mark.parametrize("atmosphere", [
        "Fine-Dining", "Casual", "Trendy", "Romantic"
    ])
    def test_atmosphere_filter_excludes_other_values(self, collection, atmosphere):
        """
        When atmosphere_filter is active, results must only contain
        restaurants with exactly that Dining Atmosphere value.
        """
        results = retrieve_restaurants(
            "dinner out",
            collection,
            top_k=5,
            atmosphere_filter=atmosphere,
        )
        for r in results:
            assert r["metadata"]["Dining Atmosphere"] == atmosphere, (
                f"Expected atmosphere='{atmosphere}', "
                f"got '{r['metadata']['Dining Atmosphere']}' "
                f"for '{r['metadata']['Name']}'"
            )

    def test_atmosphere_filter_none_does_not_restrict(self, collection):
        """
        atmosphere_filter=None must allow all atmosphere types to appear.
        """
        results = retrieve_restaurants(
            "restaurant",
            collection,
            top_k=20,
            atmosphere_filter=None,
        )
        atmospheres_found = {r["metadata"]["Dining Atmosphere"] for r in results}
        assert len(atmospheres_found) > 1, (
            f"Expected diverse atmospheres with no filter, got: {atmospheres_found}"
        )

    def test_romantic_atmosphere_result_count_capped_by_dataset(
        self, collection, df_counts
    ):
        """
        Only 3 restaurants have Dining Atmosphere == 'Romantic'.
        top_k=10 must return at most 3, not raise or return invalid data.
        """
        results = retrieve_restaurants(
            "romantic dinner",
            collection,
            top_k=10,
            atmosphere_filter="Romantic",
        )
        assert len(results) <= df_counts["atmosphere_romantic"], (
            f"Got {len(results)} Romantic results but only "
            f"{df_counts['atmosphere_romantic']} exist"
        )
        for r in results:
            assert r["metadata"]["Dining Atmosphere"] == "Romantic"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ROOFTOP FILTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRooftopFilter:

    ROOFTOP_NAMES = {
        "71Above", "La Boucherie", "Aperture at City Club LA",
        "Elephante", "Yamashiro", "LouLou"
    }

    def test_rooftop_only_returns_only_rooftop_restaurants(self, collection):
        """
        rooftop_only=True must ensure every result has Sky-High Rooftop == 'Yes'.
        """
        results = retrieve_restaurants(
            "views restaurant",
            collection,
            top_k=10,
            rooftop_only=True,
        )
        for r in results:
            assert r["metadata"]["Sky-High Rooftop"] == "Yes", (
                f"'{r['metadata']['Name']}' is not a rooftop restaurant "
                f"but appeared with rooftop_only=True"
            )

    def test_rooftop_only_result_count_capped_at_six(self, collection, df_counts):
        """
        Only 6 rooftop restaurants exist. top_k=20 must return at most 6.
        """
        results = retrieve_restaurants(
            "sky high dining",
            collection,
            top_k=20,
            rooftop_only=True,
        )
        assert len(results) <= df_counts["rooftop_yes"], (
            f"Got {len(results)} rooftop results but only {df_counts['rooftop_yes']} exist"
        )

    def test_rooftop_result_names_are_known_rooftop_restaurants(self, collection):
        """
        Every name returned by rooftop_only=True must be one of the
        six known rooftop restaurants in the dataset.
        """
        results = retrieve_restaurants(
            "panoramic views",
            collection,
            top_k=10,
            rooftop_only=True,
        )
        for r in results:
            assert r["metadata"]["Name"] in self.ROOFTOP_NAMES, (
                f"Unexpected non-rooftop restaurant in results: "
                f"'{r['metadata']['Name']}'"
            )

    def test_rooftop_false_does_not_restrict_results(self, collection):
        """
        rooftop_only=False (the default) must not restrict results to
        rooftop restaurants only — non-rooftop restaurants must appear.
        """
        results = retrieve_restaurants(
            "dinner tonight",
            collection,
            top_k=10,
            rooftop_only=False,
        )
        non_rooftop_count = sum(
            1 for r in results if r["metadata"]["Sky-High Rooftop"] == "No"
        )
        assert non_rooftop_count > 0, (
            "With rooftop_only=False, expected non-rooftop restaurants in results"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COMBINED FILTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCombinedFilters:

    def test_michelin_and_price_filter_combined(self, collection):
        """
        Both michelin_filter and price_filter active simultaneously.
        Every result must satisfy BOTH constraints.
        """
        results = retrieve_restaurants(
            "special occasion dinner",
            collection,
            top_k=5,
            michelin_filter="1-Star",
            price_filter="$$$$",
        )
        for r in results:
            assert r["metadata"]["Michelin-Guide"] == "1-Star", (
                f"Michelin filter violated: '{r['metadata']['Name']}' "
                f"has '{r['metadata']['Michelin-Guide']}'"
            )
            assert r["metadata"]["Price"] == "$$$$", (
                f"Price filter violated: '{r['metadata']['Name']}' "
                f"has '{r['metadata']['Price']}'"
            )

    def test_michelin_and_atmosphere_filter_combined(self, collection):
        """michelin_filter + atmosphere_filter: both constraints must hold."""
        results = retrieve_restaurants(
            "intimate kaiseki",
            collection,
            top_k=5,
            michelin_filter="2-Star",
            atmosphere_filter="Fine-Dining",
        )
        for r in results:
            assert r["metadata"]["Michelin-Guide"] == "2-Star"
            assert r["metadata"]["Dining Atmosphere"] == "Fine-Dining"

    def test_rooftop_and_atmosphere_filter_combined(self, collection):
        """
        rooftop_only=True + atmosphere_filter='Fine-Dining'.
        Only 71Above, La Boucherie, and Aperture at City Club LA are both
        rooftop AND Fine-Dining (3 restaurants in the real dataset).
        """
        results = retrieve_restaurants(
            "skyline views fine dining",
            collection,
            top_k=10,
            atmosphere_filter="Fine-Dining",
            rooftop_only=True,
        )
        # Both constraints must hold on every returned result
        for r in results:
            assert r["metadata"]["Sky-High Rooftop"] == "Yes", (
                f"rooftop_only violated: '{r['metadata']['Name']}'"
            )
            assert r["metadata"]["Dining Atmosphere"] == "Fine-Dining", (
                f"atmosphere filter violated: '{r['metadata']['Name']}'"
            )
        # Expect at most 3 (the real intersection of rooftop + Fine-Dining)
        assert len(results) <= 3, (
            f"Expected at most 3 results for rooftop + Fine-Dining, got {len(results)}"
        )

    def test_all_four_filters_active(self, collection):
        """
        All four filters active simultaneously must still return only
        results satisfying every constraint — even if that means 0 results.
        """
        results = retrieve_restaurants(
            "best omakase",
            collection,
            top_k=5,
            michelin_filter="1-Star",
            price_filter="$$$$",
            atmosphere_filter="Fine-Dining",
            rooftop_only=False,
        )
        for r in results:
            assert r["metadata"]["Michelin-Guide"] == "1-Star"
            assert r["metadata"]["Price"] == "$$$$"
            assert r["metadata"]["Dining Atmosphere"] == "Fine-Dining"
            assert r["metadata"]["Sky-High Rooftop"] == "No"

    def test_contradictory_filters_return_empty_list(self, collection):
        """
        Contradictory constraints (rooftop=True + atmosphere=Romantic)
        have zero matching restaurants in the real data.
        The function must return [] not raise an exception.
        """
        # Verify contradiction: no restaurant is both rooftop AND Romantic
        df = pd.read_csv(FINAL_CSV)
        impossible_count = len(
            df[(df["Sky-High Rooftop"] == "Yes") & (df["Dining Atmosphere"] == "Romantic")]
        )
        assert impossible_count == 0, (
            "Test precondition failed — a rooftop+Romantic restaurant now exists. "
            "Update this test to use a genuinely contradictory filter pair."
        )

        results = retrieve_restaurants(
            "romantic rooftop",
            collection,
            top_k=5,
            atmosphere_filter="Romantic",
            rooftop_only=True,
        )
        assert results == [], (
            f"Expected empty list for impossible filter combination, got: "
            f"{[r['metadata']['Name'] for r in results]}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TOP_K BOUNDARY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopKBoundary:

    def test_top_k_1_returns_exactly_one_result(self, collection):
        """top_k=1 must return exactly one result."""
        results = retrieve_restaurants("omakase sushi", collection, top_k=1)
        assert len(results) == 1

    def test_top_k_equals_corpus_size_does_not_raise(self, collection, df_counts):
        """
        top_k equal to the full corpus size (71) must not raise.
        Returns up to 71 results.
        """
        results = retrieve_restaurants(
            "restaurant",
            collection,
            top_k=df_counts["total"],
        )
        assert len(results) <= df_counts["total"]
        assert len(results) > 0

    def test_top_k_larger_than_corpus_does_not_raise(self, collection):
        """
        top_k=200 (larger than the 71-restaurant corpus) must not raise
        and must return at most 71 results.
        """
        results = retrieve_restaurants("restaurant", collection, top_k=200)
        assert len(results) <= 71
        assert isinstance(results, list)

    def test_custom_top_k_values_respected(self, collection):
        """top_k values of 3, 7, and 12 must return exactly that many results
        when sufficient unfiltered candidates exist."""
        for k in [3, 7, 12]:
            results = retrieve_restaurants(
                "fine dining",
                collection,
                top_k=k,
                michelin_filter=None,
            )
            assert len(results) == k, (
                f"top_k={k} returned {len(results)} results instead"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CANDIDATE POOL STRATEGY TESTS
#
# The candidate pool prevents filter depletion: the function queries for
# max(20, top_k * 4) candidates first, then filters in Python.
# This means: with a broad filter and top_k=5, we should always get 5
# results even if a naive single-pass query would have come up short.
# ═══════════════════════════════════════════════════════════════════════════════

class TestCandidatePoolStrategy:

    def test_broad_filter_returns_full_top_k(self, collection, df_counts):
        """
        michelin_filter='No' covers 24 restaurants total, but the FakeCollection
        returns corpus rows in CSV order (Michelin stars first). The candidate
        pool for top_k=15 is max(20, 15*4)=60, which contains 15 'No' restaurants.
        The function must return exactly 15 — pool is large enough.

        Note on FakeCollection pool behavior (deterministic, CSV-ordered):
          top_k=5  → pool=20 → 0 'No' restaurants in pool (all stars at top)
          top_k=10 → pool=40 → 4 'No' restaurants in pool
          top_k=15 → pool=60 → 15 'No' restaurants in pool  ← this test
        """
        results = retrieve_restaurants(
            "upscale dinner",
            collection,
            top_k=15,
            michelin_filter="No",
        )
        assert len(results) == 15, (
            f"Expected 15 results with michelin_filter='No' and pool_size=60, "
            f"got {len(results)}"
        )

    def test_fine_dining_filter_returns_full_top_k(self, collection, df_counts):
        """
        atmosphere_filter='Fine-Dining' covers 40 restaurants.
        top_k=10 must return exactly 10 results.
        """
        results = retrieve_restaurants(
            "elegant dinner",
            collection,
            top_k=10,
            atmosphere_filter="Fine-Dining",
        )
        assert len(results) == 10, (
            f"Expected 10 Fine-Dining results, got {len(results)}"
        )

    def test_pool_size_is_at_least_20(self, collection):
        """
        The candidate pool must be at least 20 (the floor).
        Verify by checking that a top_k=1 call with a common filter
        still sources from a pool large enough to satisfy the result.
        """
        # With pool_size = max(20, 1*4) = 20, a fine-dining filter
        # over 40 restaurants should always saturate the pool and return 1.
        results = retrieve_restaurants(
            "any fine dining",
            collection,
            top_k=1,
            atmosphere_filter="Fine-Dining",
        )
        assert len(results) == 1

    def test_pool_scales_with_top_k(self, collection):
        """
        Pool size = max(20, top_k * 4). With top_k=15, pool = 60.
        With michelin_filter='No' (24 results), we should still get 15
        because 60-candidate pool exceeds the filter's reach into corpus.
        """
        results = retrieve_restaurants(
            "great food",
            collection,
            top_k=15,
            michelin_filter="No",
        )
        # 24 matching restaurants exist, top_k=15 → must return 15
        assert len(results) == 15, (
            f"Expected 15 results from pool-scaled query, got {len(results)}"
        )

    def test_tight_filter_gracefully_returns_fewer_than_top_k(self, collection, df_counts):
        """
        When the dataset genuinely has fewer matches than top_k,
        the function must return all available matches, not raise or pad.

        3-Star only = 2 restaurants. top_k=5 → must return 2, not 5.
        """
        results = retrieve_restaurants(
            "world class dining",
            collection,
            top_k=5,
            michelin_filter="3-Star",
        )
        assert len(results) == df_counts["michelin_3star"], (
            f"Expected exactly {df_counts['michelin_3star']} 3-Star results "
            f"(all that exist), got {len(results)}"
        )
        assert len(results) < 5, (
            "3-Star filter should return fewer than top_k=5 since only 2 exist"
        )

    def test_rooftop_filter_gracefully_returns_all_six(self, collection, df_counts):
        """
        Only 6 rooftop restaurants exist, at CSV positions 36, 40, 62, 66, 69, 70.
        To reach all 6 the candidate pool must include the full corpus (71 rows).
        pool = max(20, top_k * 4) → top_k=18 gives pool=72, which covers all 71.

        With top_k=18 the function must return all 6 rooftop restaurants,
        not raise or return garbage to pad to 18.
        """
        results = retrieve_restaurants(
            "rooftop dining",
            collection,
            top_k=18,           # pool = max(20, 18*4) = 72 → full corpus reached
            rooftop_only=True,
        )
        assert len(results) == df_counts["rooftop_yes"], (
            f"Expected all {df_counts['rooftop_yes']} rooftop results with "
            f"full-corpus pool, got {len(results)}"
        )

    def test_no_result_exceeds_top_k_under_any_filter(self, collection):
        """
        Under any filter combination, result count must never exceed top_k.
        Tested across multiple top_k values and filter pairs.
        """
        cases = [
            dict(top_k=3, michelin_filter="1-Star"),
            dict(top_k=5, price_filter="$$$$"),
            dict(top_k=4, atmosphere_filter="Casual"),
            dict(top_k=2, rooftop_only=True),
            dict(top_k=8, michelin_filter=["1-Star", "2-Star"], price_filter="$$$$"),
        ]
        for case in cases:
            k = case.pop("top_k")
            results = retrieve_restaurants("dinner", collection, top_k=k, **case)
            assert len(results) <= k, (
                f"Result count {len(results)} exceeded top_k={k} "
                f"with filters: {case}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FILTER ISOLATION TESTS
#
# Active filters must never "bleed" — i.e., non-matching rows must never
# appear in results regardless of semantic similarity score.
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterIsolation:

    def test_michelin_filter_never_bleeds_non_matching_tier(self, collection):
        """
        After applying michelin_filter='Bib-Gourmand', zero non-Bib results
        must appear even with top_k=10.
        """
        results = retrieve_restaurants(
            "affordable Michelin meal",
            collection,
            top_k=10,
            michelin_filter="Bib-Gourmand",
        )
        non_bib = [
            r["metadata"]["Name"]
            for r in results
            if r["metadata"]["Michelin-Guide"] != "Bib-Gourmand"
        ]
        assert not non_bib, (
            f"Non-Bib-Gourmand restaurants in results: {non_bib}"
        )

    def test_price_filter_never_bleeds_adjacent_tier(self, collection):
        """
        price_filter='$$$' must never return '$$$$' or '$$' restaurants,
        even though their metadata strings all contain dollar signs.
        """
        results = retrieve_restaurants(
            "dinner",
            collection,
            top_k=10,
            price_filter="$$$",
        )
        wrong_price = [
            (r["metadata"]["Name"], r["metadata"]["Price"])
            for r in results
            if r["metadata"]["Price"] != "$$$"
        ]
        assert not wrong_price, (
            f"Price filter '$$$' returned wrong-tier results: {wrong_price}"
        )

    def test_rooftop_filter_never_returns_ground_floor_restaurants(self, collection):
        """
        rooftop_only=True must produce zero Sky-High Rooftop == 'No' results.
        """
        results = retrieve_restaurants(
            "restaurant with views",
            collection,
            top_k=10,
            rooftop_only=True,
        )
        ground_floor = [
            r["metadata"]["Name"]
            for r in results
            if r["metadata"]["Sky-High Rooftop"] != "Yes"
        ]
        assert not ground_floor, (
            f"Non-rooftop restaurants appeared with rooftop_only=True: {ground_floor}"
        )

    def test_atmosphere_filter_never_bleeds_composite_atmosphere_values(
        self, collection
    ):
        """
        atmosphere_filter='Romantic' must not return restaurants with
        composite atmospheres like 'Romantic / Smart-Casual'.
        These are distinct string values and must not match.
        """
        results = retrieve_restaurants(
            "romantic dinner",
            collection,
            top_k=10,
            atmosphere_filter="Romantic",
        )
        for r in results:
            assert r["metadata"]["Dining Atmosphere"] == "Romantic", (
                f"'{r['metadata']['Name']}' has atmosphere "
                f"'{r['metadata']['Dining Atmosphere']}', not exact 'Romantic'"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. RESULT STRUCTURE CONSISTENCY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestResultStructureConsistency:

    def test_metadata_name_appears_in_document(self, collection):
        """
        Every result's metadata['Name'] must appear somewhere in its
        document string — the metadata and document must be consistent.
        """
        results = retrieve_restaurants(
            "omakase counter",
            collection,
            top_k=10,
        )
        for r in results:
            name = r["metadata"]["Name"]
            assert name in r["document"], (
                f"Restaurant name '{name}' not found in its own document.\n"
                f"Document: {r['document'][:120]}..."
            )

    def test_metadata_price_appears_in_document(self, collection):
        """
        The price symbols from metadata must appear in the document string,
        confirming the metadata and document come from the same record.
        """
        results = retrieve_restaurants(
            "Beverly Hills dinner",
            collection,
            top_k=10,
        )
        for r in results:
            price = r["metadata"]["Price"]
            assert price in r["document"], (
                f"Price '{price}' not found in document for "
                f"'{r['metadata']['Name']}'"
            )

    def test_metadata_michelin_guide_consistent_with_document(self, collection):
        """
        The formatted Michelin label in the document must correspond to the
        raw Michelin-Guide value in metadata.
        """
        label_map = {
            "3-Star":            "Michelin 3-Star",
            "2-Star":            "Michelin 2-Star",
            "1-Star":            "Michelin 1-Star",
            "Bib-Gourmand":      "Michelin Bib-Gourmand",
            "Michelin-Selected": "Michelin-Selected",
            "No":                "No",
        }
        results = retrieve_restaurants(
            "fine dining",
            collection,
            top_k=10,
        )
        for r in results:
            raw_value     = r["metadata"]["Michelin-Guide"]
            expected_label = label_map.get(raw_value)
            if expected_label:
                assert expected_label in r["document"], (
                    f"'{r['metadata']['Name']}': expected Michelin label "
                    f"'{expected_label}' in document.\nDocument: {r['document'][:120]}"
                )
