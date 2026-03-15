# ============================================================
#  LA Luxury Restaurant Recommender — Gradio Dashboard
#  Phase 5: User-Facing Interface
#  Uses: restaurants_with_emotions.csv + ChromaDB vector store
# ============================================================

import os
import re
import warnings
import logging
import pandas as pd
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Suppress known harmless startup warnings ────────────────
# Pydantic V1 / Python 3.14 compatibility notice (not a crash)
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
warnings.filterwarnings("ignore", message=".*pydantic.v1.*")
# sentence-transformers BertModel position_ids key (harmless load-order artifact)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ── Environment & Data ─────────────────────────────────────
load_dotenv()

restaurants = pd.read_csv("../data/restaurants_with_emotions.csv")

# ── Load or Rebuild ChromaDB Vector Store ──────────────────
CHROMA_DIR   = "../chroma_restaurants"
TXT_PATH     = "../data/tagged_restaurant_descriptions.txt"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    # Load persisted database from Notebook 2
    db_restaurants = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="la_restaurants"
    )
else:
    # Build from tagged descriptions if chroma dir is missing
    raw_documents = TextLoader(TXT_PATH, encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1,
        chunk_overlap=0
    )
    documents = text_splitter.split_documents(raw_documents)
    db_restaurants = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="la_restaurants"
    )


# ── Helper: Extract Restaurant Name from Metadata ──────────
def extract_name(page_content: str) -> str:
    """Extracts restaurant name from metadata string 'Name is a ...'"""
    content = page_content.strip().strip('"')
    if " is a " in content:
        return content.split(" is a ")[0].strip()
    if " is " in content:
        return content.split(" is ")[0].strip()
    return content[:40].strip()


# ── Core Retrieval Function ─────────────────────────────────
def retrieve_semantic_recommendations(
    query:              str,
    cuisine_group:      str  = "All",
    location:           str  = "All",
    price:              str  = "All",
    michelin:           str  = "All",
    occasion:           str  = "All",
    vibe:               str  = "All",
    emotion_sort:       str  = "None",
    rooftop_only:       bool = False,
    initial_top_k:      int  = 71,
    final_top_k:        int  = 12,
) -> pd.DataFrame:
    """
    Semantic + filter restaurant retrieval pipeline.

    1. Query ChromaDB for semantic matches (pulls all 71 as candidate pool
       so downstream filters always have enough results to work with).
    2. Map doc results back to the full DataFrame by restaurant name.
    3. Apply user-selected post-retrieval filters.
    4. Optionally re-sort by a specific emotion score.
    5. Return up to final_top_k results.
    """
    # Step 1: Semantic search
    raw_docs = db_restaurants.similarity_search(query, k=initial_top_k)

    # Step 2: Map to DataFrame preserving semantic rank order
    matched_names = []
    seen = set()
    for doc in raw_docs:
        name = extract_name(doc.page_content)
        if name and name not in seen:
            matched_names.append(name)
            seen.add(name)

    results = []
    for name in matched_names:
        rows = restaurants[restaurants["Name"] == name]
        if not rows.empty:
            results.append(rows.iloc[0])

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).reset_index(drop=True)

    # Step 3: Apply filters
    if cuisine_group != "All":
        df = df[df["simple_cuisine_group"] == cuisine_group]
    if location != "All":
        df = df[df["Location"] == location]
    if price != "All":
        df = df[df["Price"] == price]
    if michelin != "All":
        df = df[df["Michelin-Guide"] == michelin]
    if occasion != "All":
        df = df[df["predicted_occasion"] == occasion]
    if vibe != "All":
        df = df[df["predicted_vibe"].str.contains(vibe, case=False, na=False)]
    if rooftop_only:
        df = df[df["Sky-High Rooftop"] == "Yes"]

    # Step 4: Re-sort by emotion score if requested
    emotion_col_map = {
        "Joy / Celebratory":      "emotion_joy",
        "Surprise / Excitement":  "emotion_surprise",
        "Passionate / Intense":   "emotion_anger",
        "Suspense / Anticipation":"emotion_fear",
        "Soulful / Reflective":   "emotion_sadness",
    }
    if emotion_sort in emotion_col_map:
        sort_col = emotion_col_map[emotion_sort]
        if sort_col in df.columns:
            df = df.sort_values(by=sort_col, ascending=False)

    return df.head(final_top_k).reset_index(drop=True)


# ── Card Renderer ───────────────────────────────────────────
MICHELIN_BADGE = {
    "3-Star":          ("⭐⭐⭐", "michelin-3"),
    "2-Star":          ("⭐⭐",   "michelin-2"),
    "1-Star":          ("⭐",     "michelin-1"),
    "Bib-Gourmand":    ("🎖️",    "michelin-bib"),
    "Michelin-Selected":("✦",    "michelin-sel"),
    "No":              ("",       ""),
}

MOOD_ICON = {
    "Dramatic & Exciting":  "🎭",
    "Elegant & Refined":    "🥂",
    "Intimate & Personal":  "🕯️",
    "Lively & Energetic":   "🎶",
    "Warm & Inviting":      "🌿",
}

PRICE_LABEL = {
    "$":     "Budget Friendly",
    "$$":    "Moderate",
    "$$$":   "Upscale",
    "$$$$":  "Fine Dining",
    "$$$$$": "Ultra Luxury",
}


def rating_stars(rating: float) -> str:
    """Convert numeric rating to filled/empty star string."""
    filled = int(round(rating))
    return "★" * filled + "☆" * (5 - filled)


def render_restaurant_card(row: pd.Series) -> str:
    """
    Renders a single restaurant as a styled HTML card string.
    All styling is inline / class-based, driven by the CSS injected
    into the Gradio Blocks head.
    """
    name        = row["Name"]
    location    = row["Location"]
    description = row["Description"]
    address     = row["Address"]
    phone       = row["Telephone Number"]
    price       = row["Price"]
    cuisine     = row["Cuisine Type"]
    atmosphere  = row["Dining Atmosphere"]
    hours       = row["Operation Hours"]
    reservations= row["Reservations"]
    dress       = row["Dress Code"]
    rating      = float(row["Customer Ratings"])
    michelin    = row["Michelin-Guide"]
    rooftop     = row["Sky-High Rooftop"] == "Yes"
    occasion    = row["predicted_occasion"]
    vibe        = row["predicted_vibe"]
    mood        = row["dining_mood"]
    fmt         = row["dining_format"]

    badge_text, badge_class = MICHELIN_BADGE.get(michelin, ("", ""))
    mood_icon   = MOOD_ICON.get(mood, "🍽️")
    price_label = PRICE_LABEL.get(price, price)
    stars_html  = rating_stars(rating)

    rooftop_badge = '<span class="badge badge-rooftop">🌆 Rooftop</span>' if rooftop else ""
    michelin_html = (
        f'<span class="badge badge-{badge_class}">{badge_text} {michelin}</span>'
        if badge_text else ""
    )
    dress_text   = {"Yes": "Dress Code Required", "Strict": "Strict Dress Code", "No": "No Dress Code"}.get(str(dress), "")
    res_icon     = {"Reservation Only": "🔒 Reservation Only", "Yes": "📅 Reservations Available", "No": "🚶 Walk-ins Welcome"}.get(str(reservations), reservations)

    return f"""
<div class="restaurant-card">
    <div class="card-header">
        <div class="card-title-row">
            <h2 class="restaurant-name">{name}</h2>
            <div class="card-badges">
                {michelin_html}
                {rooftop_badge}
            </div>
        </div>
        <div class="card-location">
            <span class="location-pin">📍</span>
            <span>{location}</span>
            <span class="separator">·</span>
            <span class="cuisine-tag">{cuisine}</span>
        </div>
    </div>

    <div class="card-rating-row">
        <span class="stars">{stars_html}</span>
        <span class="rating-num">{rating:.1f}</span>
        <span class="separator">·</span>
        <span class="price-tag" title="{price_label}">{price}</span>
        <span class="separator">·</span>
        <span class="atmosphere-tag">{atmosphere}</span>
    </div>

    <p class="card-description">{description}</p>

    <div class="card-tags">
        <span class="tag tag-occasion">🎯 {occasion}</span>
        <span class="tag tag-vibe">✨ {vibe}</span>
        <span class="tag tag-mood">{mood_icon} {mood}</span>
        <span class="tag tag-format">🍽️ {fmt}</span>
    </div>

    <div class="card-details">
        <div class="detail-item">
            <span class="detail-icon">🕐</span>
            <span>{hours}</span>
        </div>
        <div class="detail-item">
            <span class="detail-icon">📞</span>
            <span>{phone}</span>
        </div>
        <div class="detail-item">
            <span class="detail-icon">📍</span>
            <span>{address}</span>
        </div>
        <div class="detail-item">
            <span class="detail-icon">🔑</span>
            <span>{res_icon}</span>
        </div>
        <div class="detail-item">
            <span class="detail-icon">👔</span>
            <span>{dress_text if dress_text else "No Dress Code"}</span>
        </div>
    </div>
</div>
"""


def build_results_html(df: pd.DataFrame) -> str:
    """Wraps all restaurant cards into a results grid."""
    if df.empty:
        return """
<div class="no-results">
    <div class="no-results-icon">🔍</div>
    <h3>No restaurants found</h3>
    <p>Try broadening your filters or changing your search query.</p>
</div>
"""
    count = len(df)
    header = f'<div class="results-header"><span class="results-count">{count} restaurant{"s" if count != 1 else ""} found</span></div>'
    cards  = "\n".join(render_restaurant_card(row) for _, row in df.iterrows())
    return f'<div class="results-wrapper">{header}<div class="cards-grid">{cards}</div></div>'


# ── Main Search Handler ─────────────────────────────────────
def search_restaurants(
    query:        str,
    cuisine:      str,
    location:     str,
    price:        str,
    michelin:     str,
    occasion:     str,
    vibe:         str,
    emotion:      str,
    rooftop:      bool,
) -> str:
    if not query or not query.strip():
        query = "upscale Los Angeles restaurant"

    df = retrieve_semantic_recommendations(
        query         = query.strip(),
        cuisine_group = cuisine,
        location      = location,
        price         = price,
        michelin      = michelin,
        occasion      = occasion,
        vibe          = vibe,
        emotion_sort  = emotion,
        rooftop_only  = rooftop,
        final_top_k   = 12,
    )
    return build_results_html(df)


# ── Dropdown Options ────────────────────────────────────────
cuisine_choices  = ["All"] + sorted(restaurants["simple_cuisine_group"].unique())
location_choices = ["All"] + sorted(restaurants["Location"].unique())
price_choices    = ["All", "$", "$$", "$$$", "$$$$", "$$$$$"]
michelin_choices = ["All", "3-Star", "2-Star", "1-Star", "Bib-Gourmand", "Michelin-Selected"]
occasion_choices = ["All"] + sorted(restaurants["predicted_occasion"].unique())
vibe_choices     = ["All"] + sorted(restaurants["predicted_vibe"].unique())
emotion_choices  = ["None", "Joy / Celebratory", "Surprise / Excitement",
                    "Passionate / Intense", "Suspense / Anticipation", "Soulful / Reflective"]


# ── Custom CSS ──────────────────────────────────────────────
# Color psychology:
#   - Deep charcoal + near-black base  → sophistication, exclusivity
#   - Rich amber / gold accent (#C9943A)→ warmth, appetite, luxury
#   - Dusty rose highlight (#C17A6F)   → romance, indulgence
#   - Sage green detail (#7A9E7E)      → freshness, comfort, welcome
#   - Off-white cream text (#F2EDE4)   → warmth, readability, elegance
CUSTOM_CSS = """
/* ── Google Fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

/* ── CSS Variables ────────────────────────────────────── */
:root {
    --bg-base:        #111009;
    --bg-surface:     #1A1812;
    --bg-card:        #201E16;
    --bg-card-hover:  #272419;
    --bg-input:       #1E1C14;
    --border:         #2E2B1F;
    --border-light:   #3D3A2C;
    --gold:           #C9943A;
    --gold-light:     #E0B060;
    --gold-dim:       #8A6320;
    --rose:           #C17A6F;
    --sage:           #7A9E7E;
    --cream:          #F2EDE4;
    --cream-dim:      #B8B0A0;
    --cream-muted:    #7A7468;
    --font-display:   'Cormorant Garamond', Georgia, serif;
    --font-body:      'Jost', sans-serif;
    --radius-sm:      6px;
    --radius-md:      12px;
    --radius-lg:      18px;
    --shadow-card:    0 4px 24px rgba(0,0,0,0.5), 0 1px 4px rgba(0,0,0,0.3);
    --shadow-glow:    0 0 30px rgba(201,148,58,0.12);
    --transition:     all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global Reset ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Gradio Frame Overrides ───────────────────────────── */
.gradio-container {
    background: var(--bg-base) !important;
    font-family: var(--font-body) !important;
    color: var(--cream) !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}
.gradio-container .prose, .gradio-container p,
.gradio-container label, .gradio-container span {
    font-family: var(--font-body) !important;
    color: var(--cream) !important;
}
footer { display: none !important; }

/* ── Site Header ──────────────────────────────────────── */
.site-header {
    text-align: center;
    padding: 52px 24px 36px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(201,148,58,0.06) 0%, transparent 100%);
    position: relative;
    overflow: hidden;
}
.site-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 40% at 50% 0%, rgba(201,148,58,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.site-header-eyebrow {
    font-family: var(--font-body);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 14px;
    opacity: 0.85;
}
.site-header h1 {
    font-family: var(--font-display) !important;
    font-size: clamp(36px, 6vw, 68px) !important;
    font-weight: 300 !important;
    color: var(--cream) !important;
    letter-spacing: -0.01em;
    line-height: 1.1;
    margin-bottom: 14px;
}
.site-header h1 em {
    font-style: italic;
    color: var(--gold-light);
}
.site-header-sub {
    font-size: 15px;
    color: var(--cream-dim);
    font-weight: 300;
    letter-spacing: 0.04em;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Search Panel ─────────────────────────────────────── */
.search-panel {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px 32px 24px;
    margin: 28px 0 24px;
    box-shadow: var(--shadow-card);
    position: relative;
}
.search-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 80px; height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}
.search-section-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--gold-dim);
    margin-bottom: 10px;
}

/* ── Gradio Input Overrides ───────────────────────────── */
.gradio-container input[type="text"],
.gradio-container textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--cream) !important;
    font-family: var(--font-body) !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
    transition: var(--transition) !important;
    caret-color: var(--gold) !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
    border-color: var(--gold-dim) !important;
    box-shadow: 0 0 0 2px rgba(201,148,58,0.15) !important;
    outline: none !important;
}
.gradio-container select,
.gradio-container .wrap {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--cream) !important;
    font-family: var(--font-body) !important;
}

/* Dropdown label fix */
.gradio-container .block > label > span,
.gradio-container .block label span {
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--cream-dim) !important;
    margin-bottom: 6px !important;
    display: block !important;
}

/* ── Search Button ────────────────────────────────────── */
.search-btn {
    background: linear-gradient(135deg, #C9943A 0%, #A87828 100%) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: #0D0B06 !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 14px 32px !important;
    cursor: pointer !important;
    transition: var(--transition) !important;
    box-shadow: 0 2px 12px rgba(201,148,58,0.3) !important;
    width: 100% !important;
    margin-top: 8px !important;
}
.search-btn:hover {
    background: linear-gradient(135deg, #E0B060 0%, #C9943A 100%) !important;
    box-shadow: 0 4px 20px rgba(201,148,58,0.45) !important;
    transform: translateY(-1px) !important;
}
.search-btn:active { transform: translateY(0) !important; }

/* ── Checkbox ─────────────────────────────────────────── */
.gradio-container input[type="checkbox"] {
    accent-color: var(--gold) !important;
    width: 16px !important;
    height: 16px !important;
}

/* ── Divider ──────────────────────────────────────────── */
.section-divider {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 4px 0 16px;
}
.section-divider span {
    font-size: 10px !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    color: var(--gold-dim) !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
}
.section-divider::before, .section-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Results ──────────────────────────────────────────── */
.results-wrapper { padding: 4px 0 32px; }
.results-header {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.results-count {
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 500;
}
.cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 20px;
}

/* ── Restaurant Card ──────────────────────────────────── */
.restaurant-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 22px 24px 20px;
    transition: var(--transition);
    box-shadow: var(--shadow-card);
    position: relative;
    overflow: hidden;
}
.restaurant-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold-dim), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
.restaurant-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-light);
    box-shadow: var(--shadow-card), var(--shadow-glow);
    transform: translateY(-2px);
}
.restaurant-card:hover::before { opacity: 1; }

/* Card Header */
.card-header { margin-bottom: 10px; }
.card-title-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 6px;
}
.restaurant-name {
    font-family: var(--font-display) !important;
    font-size: 22px !important;
    font-weight: 400 !important;
    color: var(--cream) !important;
    line-height: 1.2 !important;
    letter-spacing: -0.01em !important;
}
.card-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    flex-shrink: 0;
    align-items: flex-start;
}
.card-location {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--cream-muted);
    letter-spacing: 0.02em;
}
.location-pin { font-size: 11px; }
.cuisine-tag {
    color: var(--sage);
    font-weight: 400;
    font-size: 11px;
}

/* Rating Row */
.card-rating-row {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}
.stars {
    color: var(--gold);
    font-size: 13px;
    letter-spacing: 1px;
}
.rating-num {
    font-size: 13px;
    font-weight: 500;
    color: var(--cream-dim);
}
.price-tag {
    font-size: 13px;
    font-weight: 500;
    color: var(--gold-light);
    letter-spacing: 0.05em;
    cursor: default;
}
.atmosphere-tag {
    font-size: 11px;
    color: var(--cream-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.separator { color: var(--border-light); font-size: 10px; }

/* Description */
.card-description {
    font-size: 13.5px;
    line-height: 1.65;
    color: var(--cream-dim);
    margin-bottom: 14px;
    font-weight: 300;
}

/* Tags Row */
.card-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 16px;
}
.tag {
    font-size: 10.5px;
    font-weight: 400;
    letter-spacing: 0.04em;
    padding: 4px 9px;
    border-radius: 20px;
    white-space: nowrap;
}
.tag-occasion { background: rgba(201,148,58,0.12); color: var(--gold-light); border: 1px solid rgba(201,148,58,0.2); }
.tag-vibe     { background: rgba(193,122,111,0.12); color: #D4907F; border: 1px solid rgba(193,122,111,0.2); }
.tag-mood     { background: rgba(122,158,126,0.12); color: var(--sage); border: 1px solid rgba(122,158,126,0.2); }
.tag-format   { background: rgba(242,237,228,0.06); color: var(--cream-muted); border: 1px solid var(--border); }

/* Detail Items */
.card-details {
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding-top: 14px;
    border-top: 1px solid var(--border);
}
.detail-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 12px;
    color: var(--cream-muted);
    line-height: 1.4;
}
.detail-icon { font-size: 11px; flex-shrink: 0; margin-top: 1px; }

/* Badges */
.badge {
    font-size: 10px;
    font-weight: 500;
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: 0.05em;
    white-space: nowrap;
}
.badge-michelin-3  { background: rgba(201,148,58,0.20); color: var(--gold-light); border: 1px solid rgba(201,148,58,0.35); }
.badge-michelin-2  { background: rgba(201,148,58,0.15); color: var(--gold); border: 1px solid rgba(201,148,58,0.25); }
.badge-michelin-1  { background: rgba(201,148,58,0.10); color: var(--gold-dim); border: 1px solid rgba(201,148,58,0.18); }
.badge-michelin-bib { background: rgba(122,158,126,0.12); color: var(--sage); border: 1px solid rgba(122,158,126,0.22); }
.badge-michelin-sel { background: rgba(242,237,228,0.06); color: var(--cream-muted); border: 1px solid var(--border); }
.badge-rooftop     { background: rgba(91,140,168,0.12); color: #85B5CF; border: 1px solid rgba(91,140,168,0.22); }

/* No Results */
.no-results {
    text-align: center;
    padding: 80px 24px;
    color: var(--cream-muted);
}
.no-results-icon { font-size: 40px; margin-bottom: 16px; opacity: 0.5; }
.no-results h3 { font-family: var(--font-display); font-size: 24px; font-weight: 300;
                 color: var(--cream-dim); margin-bottom: 10px; }
.no-results p  { font-size: 14px; }

/* ── HTML Output Block ────────────────────────────────── */
.gradio-container .output-html {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gold-dim); }
"""


# ── Build the Gradio Theme ──────────────────────────────────
# In Gradio 6.0+, theme and css moved from gr.Blocks() to .launch()
GRADIO_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.stone,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Jost"), "sans-serif"],
)

# ── Gradio Layout ───────────────────────────────────────────
with gr.Blocks(title="LA Luxury Restaurant Recommender") as dashboard:

    # ── Header ───────────────────────────────────────────
    gr.HTML("""
    <div class="site-header">
        <div class="site-header-eyebrow">Los Angeles · Curated Dining Guide</div>
        <h1>Discover <em>Extraordinary</em><br>Dining in LA</h1>
        <p class="site-header-sub">
            Semantic search across 71 curated upscale and Michelin-recognized
            restaurants — filtered by cuisine, vibe, price, and emotion.
        </p>
    </div>
    """)

    # ── Search Panel ─────────────────────────────────────
    with gr.Group(elem_classes="search-panel"):

        gr.HTML('<div class="section-divider"><span>Your Search</span></div>')

        user_query = gr.Textbox(
            label="Describe the experience you're looking for",
            placeholder="e.g., intimate omakase for a special anniversary · rooftop dinner with views · casual Michelin tacos",
            lines=2,
        )

        gr.HTML('<div class="section-divider"><span>Refine by</span></div>')

        # Row 1: Cuisine, Location, Price, Michelin
        with gr.Row():
            cuisine_dd = gr.Dropdown(
                choices=cuisine_choices,
                value="All",
                label="Cuisine",
            )
            location_dd = gr.Dropdown(
                choices=location_choices,
                value="All",
                label="City / District",
            )
            price_dd = gr.Dropdown(
                choices=price_choices,
                value="All",
                label="Price Range",
            )
            michelin_dd = gr.Dropdown(
                choices=michelin_choices,
                value="All",
                label="Michelin Rating",
            )

        # Row 2: Occasion, Vibe, Emotion Sort, Rooftop
        with gr.Row():
            occasion_dd = gr.Dropdown(
                choices=occasion_choices,
                value="All",
                label="Occasion",
            )
            vibe_dd = gr.Dropdown(
                choices=vibe_choices,
                value="All",
                label="Vibe",
            )
            emotion_dd = gr.Dropdown(
                choices=emotion_choices,
                value="None",
                label="Sort by Emotion",
            )
            rooftop_cb = gr.Checkbox(
                label="🌆  Rooftop / Sky-High Views Only",
                value=False,
            )
            
        search_btn = gr.Button(
            "✦  Find My Restaurant",
            elem_classes="search-btn",
        )

    # ── Results Output ────────────────────────────────────
    results_output = gr.HTML(
        value="""
<div class="no-results" style="padding:60px 24px">
    <div class="no-results-icon">🍽️</div>
    <h3>Ready when you are</h3>
    <p>Enter a search above to discover your perfect Los Angeles dining experience.</p>
</div>
""",
        label="",
    )

    # ── Wire Up ───────────────────────────────────────────
    search_btn.click(
        fn=search_restaurants,
        inputs=[
            user_query,
            cuisine_dd,
            location_dd,
            price_dd,
            michelin_dd,
            occasion_dd,
            vibe_dd,
            emotion_dd,
            rooftop_cb,
        ],
        outputs=results_output,
    )

    # Also fire on Enter key in the text box
    user_query.submit(
        fn=search_restaurants,
        inputs=[
            user_query,
            cuisine_dd,
            location_dd,
            price_dd,
            michelin_dd,
            occasion_dd,
            vibe_dd,
            emotion_dd,
            rooftop_cb,
        ],
        outputs=results_output,
    )


# ── Launch ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🍽️  LA Luxury Restaurant Recommender")
    print("  Open your browser and go to:")
    print("  ➜  http://localhost:7860")
    print("=" * 55 + "\n")
    dashboard.launch(
        server_name="127.0.0.1",  # Browser address: http://localhost:7860
        server_port=7860,
        share=False,               # Set True only if you need a public tunnel
        show_error=True,
        inbrowser=True,            # Auto-opens your default browser on launch
        favicon_path=None,
        theme=GRADIO_THEME,        # Gradio 6.0+: theme moves to launch()
        css=CUSTOM_CSS,            # Gradio 6.0+: css moves to launch()
    )
