from __future__ import annotations

import math
import html
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


APP_PATH = Path(__file__).resolve()
APP_DIR = APP_PATH.parent
DEPLOYMENT_ROOT = Path("/home/allisond/data_processing")
DEPLOYMENT_PIPELINE = DEPLOYMENT_ROOT / "amazon_pipeline.py"
DEFAULT_DATA_PROCESSING_DIRS = [DEPLOYMENT_ROOT]
ARTIFACT_DIRS = [
    DEPLOYMENT_ROOT / "artifacts",
    DEPLOYMENT_ROOT / "outputs",
    DEPLOYMENT_ROOT / "data",
    DEPLOYMENT_ROOT,
    APP_DIR / "artifacts",
    APP_DIR / "outputs",
    APP_DIR / "data",
    APP_DIR.parent / "artifacts",
    APP_DIR.parent / "outputs",
    APP_DIR.parent / "data",
    APP_DIR,
    APP_DIR.parent,
    Path.cwd(),
    Path.cwd().parent,
]

def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        try:
            key = str(path.expanduser().resolve())
        except Exception:
            key = str(path.expanduser())
        if key not in seen:
            seen.add(key)
            out.append(path.expanduser())
    return out


def pipeline_search_paths() -> list[Path]:
    env_path = os.environ.get("AMAZON_PIPELINE_PATH")
    ui_path = ""
    try:
        ui_path = st.session_state.get("pipeline_path_override", "")
    except Exception:
        ui_path = ""
    paths: list[Path] = []
    if env_path:
        paths.append(Path(env_path).expanduser())
    if ui_path:
        paths.append(Path(ui_path).expanduser())
    paths.append(DEPLOYMENT_PIPELINE)

    # Always search the directory containing this app.py — covers both the Linux
    # deployment root and any local copy (e.g. amazon_cuda_workbench on Mac).
    paths.append(APP_DIR / "amazon_pipeline.py")
    paths.append(APP_DIR.parent / "amazon_pipeline.py")
    return unique_paths(paths)

PRODUCT_FILES = [
    "amazon_products_clean.parquet",
    "products_clean.parquet",
    "amazon_products.parquet",
]

REVIEW_FILES = [
    "review_topic_points.parquet",
    "amazon_reviews_sentiment.parquet",
    "reviews_with_topics.parquet",
    "amazon_reviews.parquet",
]


st.set_page_config(
    page_title="CUDA-X Amazon Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --accent: #76b900;
            --accent-2: #9ce93a;
            --panel: #0a0f0a;
            --panel-2: #101610;
            --panel-3: #151d15;
            --border: #263426;
            --text: #edf3ea;
            --muted: #91a08d;
            --grid: rgba(118, 185, 0, 0.14);
        }
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background: #050705 !important;
            color: var(--text);
        }
        header[data-testid="stHeader"], [data-testid="stToolbar"] {
            background: rgba(5, 7, 5, 0) !important;
        }
        .block-container {
            max-width: 100% !important;
            padding: 0.7rem 1.1rem 1.8rem !important;
        }
        [data-testid="stSidebar"] {
            background: #0b0e0b;
            border-right: 1px solid var(--border);
        }
        h1, h2, h3, h4, h5, h6, p, span, label, div {
            color: inherit;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea,
        input {
            background: #0b110b !important;
            border-color: #2a3b2a !important;
            color: #eef8ea !important;
        }
        .stSlider [data-baseweb="slider"] div {
            color: var(--accent);
        }
        .topbar {
            display: grid;
            grid-template-columns: 1.2fr 1.8fr auto;
            align-items: center;
            gap: 14px;
            border: 1px solid #253825;
            border-radius: 8px;
            padding: 11px 13px;
            background:
                linear-gradient(90deg, rgba(118,185,0,.12), rgba(12,16,12,.94)),
                radial-gradient(circle at 60% 0%, rgba(118,185,0,.18), transparent 32%);
            box-shadow: 0 0 0 1px rgba(118,185,0,.05), 0 18px 34px rgba(0,0,0,.24);
            margin-bottom: 8px;
        }
        .brand-eyebrow {
            color: var(--accent-2);
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .brand-title {
            margin: 1px 0 0;
            color: #f6fff2;
            font-size: 22px;
            font-weight: 820;
            line-height: 1.1;
        }
        .top-meta {
            display: flex;
            justify-content: center;
            gap: 16px;
            color: var(--muted);
            font-size: 12px;
            white-space: nowrap;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            border: 1px solid rgba(118,185,0,.42);
            border-radius: 999px;
            padding: 6px 11px;
            color: #eaffdf;
            background: rgba(118,185,0,.12);
            font-size: 12px;
            font-weight: 700;
        }
        .dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: var(--accent);
            box-shadow: 0 0 10px rgba(118,185,0,.7);
        }
        .control-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 12px 2px;
            background: #090d09;
            margin-bottom: 8px;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(120px, 1fr));
            gap: 8px;
            margin: 8px 0;
        }
        .kpi {
            min-height: 66px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: linear-gradient(180deg, #0f160f, #070b07);
            padding: 10px 12px;
        }
        .kpi-label {
            color: var(--muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: .03em;
        }
        .kpi-value {
            color: #eaf7e4;
            font-size: 25px;
            font-weight: 780;
            margin-top: 4px;
            line-height: 1;
        }
        .kpi-sub {
            color: var(--accent-2);
            font-size: 11px;
            margin-top: 8px;
        }
        .stage-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(120px, 1fr));
            gap: 6px;
            margin: 4px 0 8px;
        }
        .stage {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 9px;
            background: #080d08;
            min-height: 60px;
        }
        .stage strong {
            color: #f4fff0;
            font-size: 12px;
        }
        .stage span {
            display: block;
            color: var(--muted);
            font-size: 11px;
            margin-top: 4px;
        }
        .panel-title {
            color: var(--accent-2);
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .05em;
            margin: 0 0 6px 0;
        }
        .panel-caption {
            color: var(--muted);
            font-size: 12px;
            margin: -2px 0 8px;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--border) !important;
            background: linear-gradient(180deg, #0a0f0a, #060906) !important;
            border-radius: 8px !important;
        }
        .review-list {
            height: 372px;
            overflow-y: auto;
            padding-right: 4px;
        }
        .review-card {
            border-bottom: 1px solid rgba(118,185,0,.12);
            padding: 7px 2px 8px;
        }
        .review-card strong {
            display: block;
            color: #eef8ea;
            font-size: 12px;
            line-height: 1.25;
        }
        .review-card span {
            display: inline-block;
            color: var(--muted);
            font-size: 11px;
            margin: 3px 8px 0 0;
        }
        .topic-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 8px;
            border-bottom: 1px solid rgba(118,185,0,.11);
            padding: 6px 0;
            align-items: center;
        }
        .topic-name {
            color: #eaf7e4;
            font-size: 12px;
            font-weight: 650;
        }
        .topic-stat {
            color: var(--accent-2);
            font-size: 11px;
            font-weight: 740;
        }
        .source-note {
            color: #7f8d7b;
            font-size: 11px;
            margin: 2px 0 8px;
        }
        .ok {
            color: var(--accent);
            font-weight: 700;
        }
        .tool-chip {
            display: inline-block;
            border: 1px solid #324432;
            border-radius: 999px;
            padding: 4px 10px;
            margin: 0 6px 6px 0;
            color: #dff7d4;
            background: #0c120c;
            font-size: 12px;
        }
        .small-note {
            color: var(--muted);
            font-size: 13px;
        }
        hr {
            border-color: rgba(118,185,0,.15) !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        [data-baseweb="popover"] ul[role="listbox"]:empty,
        [data-baseweb="popover"] ul[role="listbox"] > li[aria-disabled="true"],
        [data-baseweb="popover"]:has(ul[role="listbox"]:empty),
        [data-baseweb="popover"]:has(ul[role="listbox"] > li[aria-disabled="true"]:only-child),
        [data-baseweb="popover"]:not(:has(li[role="option"]:not([aria-disabled="true"]))) {
            display: none !important;
        }
        @media (max-width: 900px) {
            .topbar { grid-template-columns: 1fr; }
            .top-meta { justify-content: flex-start; flex-wrap: wrap; }
            .kpi-grid, .stage-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_artifact(candidates: list[str]) -> Path | None:
    for base in ARTIFACT_DIRS:
        for name in candidates:
            path = base / name
            if path.exists():
                return path
    return None


def find_pipeline() -> Path | None:
    for path in pipeline_search_paths():
        if path.exists():
            return path.resolve()
    return None


def pipeline_debug_text() -> str:
    lines = [
        f"expected deployment root: {DEPLOYMENT_ROOT}",
        f"expected pipeline: {DEPLOYMENT_PIPELINE}",
        f"app file: {APP_PATH}",
        f"app dir: {APP_DIR}",
        f"current working dir: {Path.cwd().resolve()}",
        f"AMAZON_PIPELINE_PATH: {os.environ.get('AMAZON_PIPELINE_PATH', '(not set)')}",
        f"UI pipeline path override: {st.session_state.get('pipeline_path_override', '(not set)') if hasattr(st, 'session_state') else '(not set)'}",
        "",
        "searched paths:",
    ]
    for path in pipeline_search_paths():
        status = "FOUND" if path.exists() else "missing"
        lines.append(f"- {status}: {path}")
    return "\n".join(lines)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in out.columns
    ]
    return out


@st.cache_data(show_spinner=False)
def demo_products(seed: int = 7, n: int = 4200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = {
        "Electronics": ["Laptops", "Headphones", "Cameras", "Storage", "Smart Home"],
        "Appliances": ["Kitchen", "Vacuum", "Laundry", "Air Quality", "Coffee"],
        "Clothing": ["Shoes", "Activewear", "Outerwear", "Accessories", "Denim"],
    }
    rows = []
    for idx in range(n):
        category = rng.choice(list(categories))
        subcategory = rng.choice(categories[category])
        list_price = float(np.round(rng.lognormal(mean=4.2, sigma=0.7), 2))
        discount_pct = float(np.clip(rng.normal(22, 14), 0, 75))
        price = float(np.round(list_price * (1 - discount_pct / 100), 2))
        rating = float(np.round(np.clip(rng.normal(4.1, 0.55), 1, 5), 2))
        rating_count = int(rng.integers(12, 25000))
        wilson_score = float(np.clip((rating / 5) * (1 - 1 / np.sqrt(rating_count)), 0, 1))
        rows.append(
            {
                "product_id": f"P{idx:06d}",
                "title": f"{subcategory} product {idx:04d}",
                "category": category,
                "subcategory": subcategory,
                "price": price,
                "list_price": list_price,
                "discount_pct": discount_pct,
                "rating": rating,
                "rating_count": rating_count,
                "wilson_score": wilson_score,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def demo_reviews(seed: int = 19, n: int = 9000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = np.array(["Electronics", "Appliances", "Clothing"])
    topics = [
        "battery life",
        "delivery experience",
        "price value",
        "product quality",
        "fit and sizing",
        "setup and usability",
        "customer service",
        "durability",
        "noise level",
        "return experience",
    ]
    rows = []
    centers = rng.normal(0, 1, size=(len(topics), 2)) * 3
    for idx in range(n):
        topic_id = int(rng.integers(0, len(topics)))
        topic_label = topics[topic_id]
        category = str(rng.choice(categories))
        sentiment_score = float(np.clip(rng.normal(0.18, 0.55), -1, 1))
        sentiment = (
            "positive"
            if sentiment_score > 0.18
            else "negative"
            if sentiment_score < -0.18
            else "neutral"
        )
        rating = int(np.clip(np.round(3.5 + sentiment_score * 1.4 + rng.normal(0, 0.8)), 1, 5))
        x, y = centers[topic_id] + rng.normal(0, 0.55, size=2)
        month = pd.Timestamp("2024-01-01") + pd.DateOffset(months=int(rng.integers(0, 24)))
        rows.append(
            {
                "review_id": f"R{idx:07d}",
                "product_id": f"P{int(rng.integers(0, 4200)):06d}",
                "category": category,
                "rating": rating,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "review_text": (
                    f"This {category.lower()} review focuses on {topic_label}. "
                    f"The customer mentions performance, value, and whether the product met expectations."
                ),
                "topic_id": topic_id,
                "topic_label": topic_label,
                "x": float(x),
                "y": float(y),
                "review_month": month,
                "similarity": float(np.clip(rng.normal(0.78, 0.12), 0.25, 0.99)),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_products() -> tuple[pd.DataFrame, str]:
    path = find_artifact(PRODUCT_FILES)
    if path is None:
        return demo_products(), "demo"
    df = pd.read_parquet(path)
    df = normalize_columns(df)
    aliases = {
        "actual_price": "price",
        "discounted_price": "price",
        "main_category": "category",
        "sub_category": "subcategory",
        "ratings": "rating",
        "no_of_ratings": "rating_count",
        "discount_percentage": "discount_pct",
    }
    df = df.rename(columns={old: new for old, new in aliases.items() if old in df.columns})
    needed = demo_products(1, 50).columns
    for col in needed:
        if col not in df.columns:
            if col in {"category", "subcategory", "title", "product_id"}:
                df[col] = "Unknown"
            else:
                df[col] = np.nan
    return df[list(needed)].copy(), str(path)


@st.cache_data(show_spinner=False)
def load_reviews() -> tuple[pd.DataFrame, str]:
    path = find_artifact(REVIEW_FILES)
    if path is None:
        return demo_reviews(), "demo"
    df = pd.read_parquet(path)
    df = normalize_columns(df)
    aliases = {
        "text": "review_text",
        "review": "review_text",
        "score": "rating",
        "topic": "topic_label",
        "umap_x": "x",
        "umap_y": "y",
        "month": "review_month",
        "date": "review_month",
    }
    df = df.rename(columns={old: new for old, new in aliases.items() if old in df.columns})
    fallback = demo_reviews(2, 100)
    for col in fallback.columns:
        if col not in df.columns:
            df[col] = fallback[col].iloc[0] if len(fallback) else None
    df["review_month"] = pd.to_datetime(df["review_month"], errors="coerce")
    return df[fallback.columns].copy(), str(path)


def run_pipeline(mode: str) -> dict[str, str | float | bool]:
    pipeline = find_pipeline()
    if pipeline is None:
        return {
            "ok": False,
            "elapsed": 0.0,
            "command": "amazon_pipeline.py not found",
            "output": (
                "The app could not locate amazon_pipeline.py.\n\n"
                "This app is configured for deployment under /home/allisond/data_processing. "
                "Run it from that directory or set AMAZON_PIPELINE_PATH to the exact pipeline file.\n\n"
                + pipeline_debug_text()
            ),
        }

    if mode == "GPU":
        command = [sys.executable, "-m", "cudf.pandas", "-m", "cuml.accel", str(pipeline)]
    else:
        command = [sys.executable, str(pipeline)]

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=pipeline.parent,
            text=True,
            capture_output=True,
            timeout=7200,
            check=False,
        )
        elapsed = time.perf_counter() - start
        output = "\n".join([proc.stdout[-2500:], proc.stderr[-2500:]]).strip()
        return {
            "ok": proc.returncode == 0,
            "elapsed": elapsed,
            "command": " ".join(command),
            "output": output or "Pipeline finished with no captured output.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed": time.perf_counter() - start,
            "command": " ".join(command),
            "output": str(exc),
        }


def filter_products(df: pd.DataFrame, categories: list[str], min_rating: float, min_discount: float, price_range: tuple[float, float], queries: list[str]) -> pd.DataFrame:
    out = df.copy()
    if categories:
        out = out[out["category"].isin(categories)]
    out = out[out["rating"].fillna(0) >= min_rating]
    out = out[out["discount_pct"].fillna(0) >= min_discount]
    out = out[out["price"].fillna(0).between(price_range[0], price_range[1])]
    terms = [q.strip().lower() for q in queries if q and q.strip()]
    if terms:
        text = out["title"].astype(str).str.lower()
        mask = pd.Series(False, index=out.index)
        for term in terms:
            mask = mask | text.str.contains(term, na=False, regex=False)
        out = out[mask]
    return out


def search_reviews(df: pd.DataFrame, categories: list[str], query: str, top_k: int, min_similarity: float, sentiment_filter: list[str]) -> pd.DataFrame:
    out = df.copy()
    if categories:
        out = out[out["category"].isin(categories)]
    if sentiment_filter:
        out = out[out["sentiment"].isin(sentiment_filter)]
    if query.strip():
        terms = [t for t in query.lower().split() if t]
        if terms:
            score = np.zeros(len(out))
            text = out["review_text"].astype(str).str.lower()
            topic = out["topic_label"].astype(str).str.lower()
            for term in terms:
                score += text.str.contains(term, regex=False).astype(float).to_numpy()
                score += 1.5 * topic.str.contains(term, regex=False).astype(float).to_numpy()
            out = out.assign(query_score=score + out["similarity"].fillna(0))
        else:
            out = out.assign(query_score=out["similarity"].fillna(0))
    else:
        out = out.assign(query_score=out["similarity"].fillna(0))
    out = out[out["similarity"].fillna(0) >= min_similarity]
    return out.sort_values("query_score", ascending=False).head(top_k)


def stage_cards(mode: str, products: pd.DataFrame, reviews: pd.DataFrame) -> None:
    multiplier = 1.0 if mode == "CPU" else 0.16
    stages = [
        ("Load Products", f"{len(products):,} rows", 12 * multiplier),
        ("Clean Structured Data", "cuDF-style transforms", 28 * multiplier),
        ("Load Reviews", f"{len(reviews):,} reviews", 18 * multiplier),
        ("Vector Search", "embeddings + retrieval", 34 * multiplier),
        ("Topic Modeling", "UMAP + HDBSCAN", 52 * multiplier),
        ("Dashboard Artifacts", "charts + tables", 9 * multiplier),
    ]
    html = "<div class='stage-grid'>"
    for name, detail, seconds in stages:
        html += (
            "<div class='stage'>"
            f"<strong>{name}</strong>"
            f"<span>{detail}</span>"
            f"<span class='ok'>{seconds:.1f}s estimated</span>"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def metric_row(products: pd.DataFrame, reviews: pd.DataFrame, mode: str, product_source: str, review_source: str) -> None:
    topics = reviews["topic_label"].nunique()
    avg_rating = products["rating"].mean()
    avg_sentiment = reviews["sentiment_score"].mean()
    speedup = "6.2x" if mode == "GPU" else "1.0x"
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Products", f"{len(products):,}")
    c2.metric("Reviews", f"{len(reviews):,}")
    c3.metric("Topics", f"{topics:,}")
    c4.metric("Avg Rating", f"{avg_rating:.2f}")
    c5.metric("Mode Speedup", speedup, delta="GPU path" if mode == "GPU" else "CPU baseline")
    st.caption(f"Product source: {product_source} | Review source: {review_source} | Avg sentiment: {avg_sentiment:.2f}")


def product_panel(products: pd.DataFrame) -> None:
    st.subheader("Structured Product Explorer")
    c1, c2 = st.columns([1.25, 1])
    with c1:
        fig = px.scatter(
            products.sample(min(len(products), 1500), random_state=2) if len(products) else products,
            x="discount_pct",
            y="rating",
            size="rating_count",
            color="category",
            hover_data=["title", "price", "subcategory"],
            title="Discount vs Rating",
            template="plotly_dark",
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        hist = px.histogram(
            products,
            x="rating",
            color="category",
            nbins=24,
            title="Rating Distribution",
            template="plotly_dark",
        )
        hist.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(hist, use_container_width=True)

    top_deals = products.sort_values(["discount_pct", "rating_count"], ascending=False).head(15)
    best_quality = products.sort_values("wilson_score", ascending=False).head(15)
    t1, t2 = st.columns(2)
    t1.dataframe(
        top_deals[["title", "category", "price", "discount_pct", "rating", "rating_count"]],
        use_container_width=True,
        hide_index=True,
    )
    t2.dataframe(
        best_quality[["title", "category", "price", "rating", "rating_count", "wilson_score"]],
        use_container_width=True,
        hide_index=True,
    )


def review_panel(reviews: pd.DataFrame, search_results: pd.DataFrame) -> None:
    st.subheader("Unstructured Review Search")
    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.caption("Top semantic-style review matches")
        display = search_results[
            ["review_text", "category", "rating", "sentiment", "topic_label", "similarity"]
        ].copy()
        display["similarity"] = display["similarity"].map(lambda x: f"{x:.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True, height=360)
    with c2:
        sample = reviews.sample(min(len(reviews), 2500), random_state=3) if len(reviews) else reviews
        fig = px.scatter(
            sample,
            x="x",
            y="y",
            color="topic_label",
            hover_data=["category", "rating", "sentiment"],
            title="Review Embedding and Topic Map",
            template="plotly_dark",
        )
        fig.update_traces(marker=dict(size=5, opacity=0.72))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=45, b=10), legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    topic_summary = (
        reviews.groupby("topic_label", dropna=False)
        .agg(
            reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
            positive_share=("sentiment", lambda s: (s == "positive").mean()),
        )
        .reset_index()
        .sort_values("reviews", ascending=False)
    )
    topic_summary["avg_rating"] = topic_summary["avg_rating"].round(2)
    topic_summary["avg_sentiment"] = topic_summary["avg_sentiment"].round(2)
    topic_summary["positive_share"] = (topic_summary["positive_share"] * 100).round(1)
    st.dataframe(topic_summary, use_container_width=True, hide_index=True)


def sentiment_panel(reviews: pd.DataFrame) -> None:
    st.subheader("Sentiment Timeline")
    timeline = (
        reviews.dropna(subset=["review_month"])
        .assign(review_month=lambda df: pd.to_datetime(df["review_month"]).dt.to_period("M").dt.to_timestamp())
        .groupby(["review_month", "category"], as_index=False)
        .agg(avg_sentiment=("sentiment_score", "mean"), reviews=("review_id", "count"))
    )
    if timeline.empty:
        st.info("No review dates available for sentiment timeline.")
        return
    fig = px.line(
        timeline,
        x="review_month",
        y="avg_sentiment",
        color="category",
        markers=True,
        hover_data=["reviews"],
        title="Average Customer Sentiment Over Time",
        template="plotly_dark",
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)


def correlation_panel(products: pd.DataFrame, reviews: pd.DataFrame) -> None:
    st.subheader("Correlation Lab")
    product_summary = (
        products.groupby("category", as_index=False)
        .agg(avg_discount=("discount_pct", "mean"), avg_product_rating=("rating", "mean"), products=("product_id", "count"))
    )
    review_summary = (
        reviews.groupby("category", as_index=False)
        .agg(avg_sentiment=("sentiment_score", "mean"), avg_review_rating=("rating", "mean"), reviews=("review_id", "count"))
    )
    merged = product_summary.merge(review_summary, on="category", how="inner")
    if merged.empty:
        st.info("Not enough category overlap to compute correlations.")
        return
    corr = merged["avg_discount"].corr(merged["avg_sentiment"])
    label = "n/a" if pd.isna(corr) else f"{corr:.2f}"
    fig = px.scatter(
        merged,
        x="avg_discount",
        y="avg_sentiment",
        size="reviews",
        color="category",
        hover_data=["avg_product_rating", "avg_review_rating", "products", "reviews"],
        title=f"Category Discount vs Review Sentiment | correlation {label}",
        template="plotly_dark",
    )
    if len(merged) >= 2:
        x = merged["avg_discount"].to_numpy(dtype=float)
        y = merged["avg_sentiment"].to_numpy(dtype=float)
        if np.isfinite(x).all() and np.isfinite(y).all() and np.nanstd(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 50)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=slope * xs + intercept,
                    mode="lines",
                    name="trend",
                    line=dict(color="#76b900", width=2),
                )
            )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)


def dark_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="#070b07",
        plot_bgcolor="#070b07",
        font=dict(color="#dce8d6", size=11),
        title=dict(font=dict(color="#eaf7e4", size=13), x=0.01),
        margin=dict(l=8, r=8, t=34, b=8),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#a9b7a3"),
            orientation="v",
        ),
        xaxis=dict(gridcolor="rgba(118,185,0,.12)", zerolinecolor="rgba(118,185,0,.18)"),
        yaxis=dict(gridcolor="rgba(118,185,0,.12)", zerolinecolor="rgba(118,185,0,.18)"),
    )
    return fig


def render_topbar(mode: str, pipeline: Path | None) -> None:
    status = "Pipeline linked" if pipeline else "Demo data mode"
    detail = str(pipeline) if pipeline else "No pipeline detected"
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="brand-eyebrow">CUDA-X Data Processing Demo</div>
                <div class="brand-title">Amazon Product + Review Intelligence</div>
            </div>
            <div class="top-meta">
                <span>Structured: Polars / Presto</span>
                <span>Unstructured: Vector Search / BERTopic</span>
                <span>ML: UMAP / HDBSCAN</span>
            </div>
            <div class="status-pill"><span class="dot"></span>{html.escape(mode)} | {html.escape(status)}</div>
        </div>
        <div class="source-note">{html.escape(detail)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_strip(products: pd.DataFrame, reviews: pd.DataFrame, mode: str) -> None:
    topics = int(reviews["topic_label"].nunique()) if len(reviews) else 0
    avg_rating = products["rating"].mean() if len(products) else np.nan
    avg_sentiment = reviews["sentiment_score"].mean() if len(reviews) else np.nan
    speedup = "6.2x" if mode == "GPU" else "1.0x"
    kpis = [
        ("Products", f"{len(products):,}", "structured rows"),
        ("Reviews", f"{len(reviews):,}", "unstructured text"),
        ("Topics", f"{topics:,}", "BERTopic clusters"),
        ("Avg Rating", f"{avg_rating:.2f}" if not pd.isna(avg_rating) else "n/a", "filtered products"),
        ("Mode Speedup", speedup, "GPU path" if mode == "GPU" else "CPU baseline"),
    ]
    cards = "".join(
        "<div class='kpi'>"
        f"<div class='kpi-label'>{html.escape(label)}</div>"
        f"<div class='kpi-value'>{html.escape(value)}</div>"
        f"<div class='kpi-sub'>{html.escape(sub)}</div>"
        "</div>"
        for label, value, sub in kpis
    )
    st.markdown(f"<div class='kpi-grid'>{cards}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='source-note'>Average sentiment: {avg_sentiment:.2f}</div>"
        if not pd.isna(avg_sentiment)
        else "<div class='source-note'>Average sentiment: n/a</div>",
        unsafe_allow_html=True,
    )


def topic_summary_frame(reviews: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty:
        return pd.DataFrame(columns=["topic_label", "reviews", "avg_rating", "avg_sentiment", "positive_share"])
    topic_summary = (
        reviews.groupby("topic_label", dropna=False)
        .agg(
            reviews=("review_id", "count"),
            avg_rating=("rating", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
            positive_share=("sentiment", lambda s: (s == "positive").mean()),
        )
        .reset_index()
        .sort_values("reviews", ascending=False)
    )
    topic_summary["avg_rating"] = topic_summary["avg_rating"].round(2)
    topic_summary["avg_sentiment"] = topic_summary["avg_sentiment"].round(2)
    topic_summary["positive_share"] = (topic_summary["positive_share"] * 100).round(1)
    return topic_summary


def render_review_cards(search_results: pd.DataFrame) -> None:
    if search_results.empty:
        st.info("No matching reviews for the current filters.")
        return
    cards = []
    for _, row in search_results.head(14).iterrows():
        text = html.escape(str(row.get("review_text", ""))[:150])
        category = html.escape(str(row.get("category", "")))
        topic = html.escape(str(row.get("topic_label", "")))
        sentiment = html.escape(str(row.get("sentiment", "")))
        rating = row.get("rating", "")
        similarity = row.get("similarity", np.nan)
        sim_text = f"{float(similarity):.2f}" if pd.notna(similarity) else "n/a"
        cards.append(
            f'<div style="border-bottom:1px solid rgba(118,185,0,.12);padding:7px 2px 8px">'
            f'<b style="display:block;color:#eef8ea;font-size:12px;line-height:1.25">{text}</b>'
            f'<span style="display:inline-block;color:#91a08d;font-size:11px;margin:3px 8px 0 0">{category}</span>'
            f'<span style="display:inline-block;color:#91a08d;font-size:11px;margin:3px 8px 0 0">{topic}</span>'
            f'<span style="display:inline-block;color:#91a08d;font-size:11px;margin:3px 8px 0 0">{sentiment}</span>'
            f'<span style="display:inline-block;color:#91a08d;font-size:11px;margin:3px 8px 0 0">rating {rating}</span>'
            f'<span style="display:inline-block;color:#91a08d;font-size:11px;margin:3px 8px 0 0">sim {sim_text}</span>'
            f'</div>'
        )
    st.markdown(
        '<div style="height:372px;overflow-y:auto;padding-right:4px">'
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_topic_rows(topic_summary: pd.DataFrame) -> None:
    if topic_summary.empty:
        st.info("No topics for the current filters.")
        return
    rows = []
    for _, row in topic_summary.iterrows():
        name = html.escape(str(row["topic_label"]))
        reviews = int(row["reviews"])
        sentiment = float(row["avg_sentiment"])
        rows.append(
            f"""
            <div class="topic-row">
                <div>
                    <div class="topic-name">{name}</div>
                    <div class="source-note">sentiment {sentiment:.2f} | rating {float(row["avg_rating"]):.2f}</div>
                </div>
                <div class="topic-stat">{reviews:,}</div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def product_scatter(products: pd.DataFrame) -> go.Figure:
    sample = products.copy()
    sample["rating_count"] = sample["rating_count"].fillna(1).clip(lower=1)
    fig = px.scatter(
        sample,
        x="discount_pct",
        y="rating",
        size="rating_count",
        color="category",
        hover_data=["title", "price", "subcategory"],
        title="Product Value Map: Discount vs Rating",
        template="plotly_dark",
        color_discrete_sequence=["#76b900", "#00bcd4", "#f7b731", "#ff5c5c", "#9b59ff"],
    )
    fig.update_traces(marker=dict(opacity=0.72, line=dict(width=0)))
    return dark_fig(fig, 350)


def rating_histogram(products: pd.DataFrame) -> go.Figure:
    data = products.copy()
    data["rating"] = data["rating"].round(1)
    fig = px.histogram(
        data,
        y="rating",
        color="category",
        title="Rating Distribution",
        template="plotly_dark",
        color_discrete_sequence=["#76b900", "#00bcd4", "#f7b731", "#ff5c5c", "#9b59ff"],
    )
    fig.update_traces(ybins=dict(start=1.0, end=5.05, size=0.1))
    fig.update_layout(xaxis_title="count", yaxis_title="rating", bargap=0.15)
    return dark_fig(fig, 320)


def topic_scatter(reviews: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        reviews,
        x="x",
        y="y",
        color="topic_label",
        hover_data=["category", "rating", "sentiment"],
        title="Review Embedding Map",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.74, line=dict(width=0)))
    return dark_fig(fig, 350)


def load_datamap_html() -> str | None:
    candidates = [
        DEPLOYMENT_ROOT / "data" / "amazon_reviews" / "bertopic_document_datamap.html",
        APP_DIR / "data" / "amazon_reviews" / "bertopic_document_datamap.html",
        APP_DIR.parent / "data" / "amazon_reviews" / "bertopic_document_datamap.html",
    ]
    for path in candidates:
        if path.exists():
            try:
                return path.read_text()
            except Exception:
                continue
    return None


def review_volume_figure(
    products: pd.DataFrame,
    reviews: pd.DataFrame,
    categories: list[str],
    categories_all: list[str],
) -> go.Figure | None:
    if reviews.empty or products.empty:
        return None
    if not categories or set(categories) == set(categories_all):
        group_col = "prod_category"
        label = "Category"
    else:
        group_col = "prod_subcategory"
        label = "Subcategory"
    prod_lookup = (
        products.drop_duplicates("product_id")
        .set_index("product_id")[["rating_count", "category", "subcategory"]]
        .rename(columns={"category": "prod_category", "subcategory": "prod_subcategory"})
    )
    merged = reviews.merge(prod_lookup, left_on="product_id", right_index=True, how="left")
    if categories and set(categories) != set(categories_all):
        merged = merged[merged["prod_category"].isin(categories)]
    merged = merged.dropna(subset=["review_month", "rating_count", group_col])
    if merged.empty:
        return None
    n_reviews_per_product = merged.groupby("product_id").size().rename("n_reviews_per_product")
    merged = merged.merge(n_reviews_per_product, left_on="product_id", right_index=True, how="left")
    merged["volume"] = merged["rating_count"] / merged["n_reviews_per_product"].clip(lower=1)
    merged["month"] = pd.to_datetime(merged["review_month"]).dt.to_period("M").dt.to_timestamp()
    grouped = (
        merged.groupby(["month", group_col], as_index=False)["volume"]
        .sum()
        .sort_values("month")
    )
    fig = px.bar(
        grouped,
        x="month",
        y="volume",
        color=group_col,
        title=f"Review Volume by Month (per {label})",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        yaxis_title="Review volume (rating-count weighted)",
        barmode="group",
    )
    return dark_fig(fig, 350)


def sentiment_figure(reviews: pd.DataFrame) -> go.Figure | None:
    timeline = (
        reviews.dropna(subset=["review_month"])
        .assign(review_month=lambda df: pd.to_datetime(df["review_month"]).dt.to_period("M").dt.to_timestamp())
        .groupby(["review_month", "category"], as_index=False)
        .agg(avg_sentiment=("sentiment_score", "mean"), reviews=("review_id", "count"))
    )
    if timeline.empty:
        return None
    fig = px.line(
        timeline,
        x="review_month",
        y="avg_sentiment",
        color="category",
        markers=True,
        hover_data=["reviews"],
        title="Sentiment Timeline",
        template="plotly_dark",
        color_discrete_sequence=["#76b900", "#00bcd4", "#f7b731", "#ff5c5c"],
    )
    return dark_fig(fig, 320)


def correlation_figure(products: pd.DataFrame, reviews: pd.DataFrame) -> go.Figure | None:
    product_summary = (
        products.groupby("category", as_index=False)
        .agg(avg_discount=("discount_pct", "mean"), avg_product_rating=("rating", "mean"), products=("product_id", "count"))
    )
    review_summary = (
        reviews.groupby("category", as_index=False)
        .agg(avg_sentiment=("sentiment_score", "mean"), avg_review_rating=("rating", "mean"), reviews=("review_id", "count"))
    )
    merged = product_summary.merge(review_summary, on="category", how="inner")
    if merged.empty:
        return None
    corr = merged["avg_discount"].corr(merged["avg_sentiment"])
    label = "n/a" if pd.isna(corr) else f"{corr:.2f}"
    fig = px.scatter(
        merged,
        x="avg_discount",
        y="avg_sentiment",
        size="reviews",
        color="category",
        hover_data=["avg_product_rating", "avg_review_rating", "products", "reviews"],
        title=f"Correlation: Discount vs Sentiment | r={label}",
        template="plotly_dark",
        color_discrete_sequence=["#76b900", "#00bcd4", "#f7b731", "#ff5c5c"],
    )
    if len(merged) >= 2:
        x = merged["avg_discount"].to_numpy(dtype=float)
        y = merged["avg_sentiment"].to_numpy(dtype=float)
        if np.isfinite(x).all() and np.isfinite(y).all() and np.nanstd(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 50)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=slope * xs + intercept,
                    mode="lines",
                    name="trend",
                    line=dict(color="#76b900", width=2),
                )
            )
    return dark_fig(fig, 320)


def top_deals_frame(products: pd.DataFrame) -> pd.DataFrame:
    cols = ["title", "category", "price", "discount_pct", "rating", "rating_count"]
    if products.empty:
        return pd.DataFrame(columns=cols)
    return products.sort_values(["discount_pct", "rating_count"], ascending=False).head(12)[cols]


def main() -> None:
    inject_css()

    products_raw, product_source = load_products()
    reviews_raw, review_source = load_reviews()

    if "pipeline_path_override" not in st.session_state:
        st.session_state["pipeline_path_override"] = os.environ.get("AMAZON_PIPELINE_PATH", "")

    categories_all = sorted(
        set(products_raw["category"].dropna().astype(str)).union(
            set(reviews_raw["category"].dropna().astype(str))
        )
    )
    max_price = float(np.nanpercentile(products_raw["price"].fillna(0), 98)) if len(products_raw) else 1000.0

    mode_default = st.session_state.get("mode", "GPU")
    mode = mode_default
    render_topbar(mode, find_pipeline())

    with st.sidebar:
        st.markdown("<div class='panel-title'>Control Surface</div>", unsafe_allow_html=True)
        mode = st.radio("Engine", ["GPU", "CPU"], horizontal=True, key="mode")
        run = st.button("Run Pipeline", type="primary", use_container_width=True)
        categories = st.multiselect("Category", categories_all, default=categories_all[:3])
        if categories:
            relevant_products = products_raw[products_raw["category"].isin(categories)]
            relevant_reviews = reviews_raw[reviews_raw["category"].isin(categories)]
        else:
            relevant_products = products_raw
            relevant_reviews = reviews_raw
        product_query_options = sorted(
            {
                str(s).strip().lower()
                for s in relevant_products["subcategory"].dropna().unique()
                if str(s).strip() and str(s).strip().lower() != "unknown"
            }
        )
        product_query = st.multiselect(
            "Product keyword",
            product_query_options,
            default=[],
            placeholder="All products",
        )
        review_query_options = sorted(
            {
                str(s).strip()
                for s in relevant_reviews["topic_label"].dropna().unique()
                if str(s).strip()
            }
        )
        review_query = st.multiselect(
            "Review search",
            review_query_options,
            default=[],
            placeholder="All reviews",
        )
        min_rating = st.slider("Min rating", 1.0, 5.0, 3.5, 0.1)
        min_discount = st.slider("Min discount", 0.0, 80.0, 10.0, 1.0)
        price_range = st.slider("Price range", 0.0, max(max_price, 1.0), (0.0, max(max_price, 1.0)), 1.0)
        min_similarity = st.slider("Similarity", 0.0, 1.0, 0.45, 0.05)
        sentiment_filter = st.multiselect(
            "Sentiment",
            ["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"],
        )

    if run:
        with st.status("Running pipeline", expanded=True) as status:
            result = run_pipeline(mode)
            if result["ok"]:
                status.update(label=f"Pipeline finished in {result['elapsed']:.1f}s", state="complete")
            else:
                status.update(label="Pipeline did not run", state="error")
            st.caption(result["command"])
            st.code(str(result["output"])[-3500:] or "No output", language="text")

    products = filter_products(products_raw, categories, min_rating, min_discount, price_range, product_query)
    reviews = reviews_raw[reviews_raw["category"].isin(categories)] if categories else reviews_raw
    if product_query:
        viz_reviews = reviews[reviews["product_id"].isin(products["product_id"])]
    else:
        viz_reviews = reviews
    if review_query:
        topics_reviews = viz_reviews[viz_reviews["topic_label"].isin(review_query)]
    else:
        topics_reviews = viz_reviews
    review_query_str = " ".join(review_query) if review_query else ""
    search_results = search_reviews(reviews, categories, review_query_str, len(reviews), min_similarity, sentiment_filter)
    topic_summary = topic_summary_frame(topics_reviews)

    render_kpi_strip(products, reviews, mode)
    stage_cards(mode, products, reviews)
    st.markdown(
        f"<div class='source-note'>Product source: {html.escape(product_source)} | Review source: {html.escape(review_source)}</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("<div class='panel-title'>Review Volume by Month</div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-caption'>Rating-count weighted monthly review volume — by category when none/all are selected, by subcategory when drilling down.</div>", unsafe_allow_html=True)
        volume_fig = review_volume_figure(products_raw, viz_reviews, categories, categories_all)
        if volume_fig is not None:
            st.plotly_chart(volume_fig, use_container_width=True)
        else:
            st.info("Not enough data to compute monthly review volume.")

    row2_left, row2_mid, row2_right = st.columns([1.0, 1.6, 1.1], gap="small")
    with row2_left:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Review Search</div>", unsafe_allow_html=True)
            st.markdown("<div class='panel-caption'>Semantic-style retrieval over unstructured customer text.</div>", unsafe_allow_html=True)
            render_review_cards(search_results)
    with row2_mid:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Topic Modeling - Document Datamap</div>", unsafe_allow_html=True)
            st.markdown("<div class='panel-caption'>Interactive cluster map produced by the pipeline — drag to pan, scroll to zoom, hover to read.</div>", unsafe_allow_html=True)
            datamap_html = load_datamap_html()
            if datamap_html is not None:
                components.html(datamap_html, height=580, scrolling=False)
            else:
                st.info("Document datamap not found. Click 'Run Pipeline' to generate it.")
    with row2_right:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Embedding and Topics</div>", unsafe_allow_html=True)
            st.markdown("<div class='panel-caption'>UMAP/HDBSCAN-style cluster map for review themes.</div>", unsafe_allow_html=True)
            st.plotly_chart(topic_scatter(topics_reviews), use_container_width=True)
            render_topic_rows(topic_summary)

    bottom_left, bottom_mid, bottom_right = st.columns(3, gap="small")
    with bottom_left:
        with st.container(border=True):
            sentiment = sentiment_figure(viz_reviews)
            if sentiment is not None:
                st.plotly_chart(sentiment, use_container_width=True)
            else:
                st.info("No review dates available for sentiment timeline.")
    with bottom_mid:
        with st.container(border=True):
            st.plotly_chart(rating_histogram(products), use_container_width=True)
    with bottom_right:
        with st.container(border=True):
            corr = correlation_figure(products, viz_reviews)
            if corr is not None:
                st.plotly_chart(corr, use_container_width=True)
            else:
                st.info("Not enough category overlap to compute correlations.")


if __name__ == "__main__":
    main()
