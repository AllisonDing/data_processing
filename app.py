"""Streamlit dashboard for the Amazon products + reviews pipelines.

Run with:
    ./run.sh                 # executes notebooks 01 & 02, then launches streamlit
    streamlit run app.py     # dashboard only (assumes artifacts already on disk)
"""
from pathlib import Path

import polars as pl
import streamlit as st
from streamlit.components.v1 import html as st_html

ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data"
PRODUCTS_DIR  = DATA_DIR / "amazon"
REVIEWS_DIR   = DATA_DIR / "amazon_reviews"

PRODUCTS_PARQUET  = PRODUCTS_DIR / "amazon_products_clean.parquet"
SENTIMENT_PARQUET = REVIEWS_DIR  / "amazon_reviews_sentiment.parquet"

st.set_page_config(
    page_title="Amazon Data Pipelines",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Amazon Data Pipelines — Dashboard")
st.caption(
    "Static showcase of the structured products + unstructured reviews "
    "pipelines from notebooks 01 & 02."
)


@st.cache_data
def load_products() -> pl.DataFrame | None:
    if not PRODUCTS_PARQUET.exists():
        return None
    return pl.read_parquet(PRODUCTS_PARQUET)


@st.cache_data
def load_sentiment() -> pl.DataFrame | None:
    if not SENTIMENT_PARQUET.exists():
        return None
    return pl.read_parquet(SENTIMENT_PARQUET)


def show_image(path: Path, caption: str = "") -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing artifact `{path.name}` — re-run the notebook to generate it.")


STATIC_DIR            = ROOT / "static"
MAX_INLINE_HTML_BYTES = 20 * 1024 * 1024   # bigger than this and srcdoc chokes


def show_html(path: Path, height: int = 700, static_url: str | None = None) -> None:
    """Embed a saved plotly HTML.

    Small files (< 20 MB) are read into a srcdoc iframe inline. Large files
    must be served via Streamlit's static-file serving (./static -> /app/static)
    — pass `static_url` so we render an iframe pointing at that URL instead of
    base64-embedding the whole blob in the page.
    """
    if not path.exists():
        st.info(f"Missing artifact `{path.name}` — re-run the notebook to generate it.")
        return
    size = path.stat().st_size
    if static_url is not None:
        st_html(
            f'<iframe src="{static_url}" width="100%" height="{height}" '
            f'style="border:none"></iframe>',
            height=height,
        )
        return
    if size > MAX_INLINE_HTML_BYTES:
        st.warning(
            f"`{path.name}` is {size / 1e6:.0f} MB — too large to embed inline. "
            f"Symlink it into `./static/` and pass `static_url` to stream via iframe."
        )
        return
    st_html(path.read_text(encoding="utf-8"), height=height, scrolling=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Overview",
    "Products (structured)",
    "Reviews EDA",
    "Sentiment Timeline",
    "Embedding Space",
    "Topics (BERTopic)",
    "Document Datamap",
])
(
    tab_overview, tab_products, tab_eda, tab_sent,
    tab_embed, tab_topics, tab_datamap,
) = tabs


with tab_overview:
    st.subheader("Project structure")
    st.markdown(
        """
| Notebook | Source | Pipeline | Output |
|---|---|---|---|
| `01_structured_amazon_products.ipynb` | Kaggle Amazon Products (139 CSVs) | Polars + cuDF GPU aggregations, Wilson rating score | `amazon_products_clean.parquet` |
| `02_unstructured_amazon_reviews.ipynb` | McAuley Lab Amazon Reviews 2023 | Embedding (Nemotron 1B), Milvus vector store, BERTopic clustering | sentiment parquet + topic HTMLs |

Every chart below was produced by running those notebooks end-to-end.
        """
    )

    products  = load_products()
    sentiment = load_sentiment()
    bertopic_count = sum(1 for _ in REVIEWS_DIR.glob("bertopic_*.html"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Products (rows)",        f"{products.height:,}"  if products  is not None else "—")
    c2.metric("Sentiment bins (months)", f"{sentiment.height:,}" if sentiment is not None else "—")
    c3.metric("BERTopic HTML panels",    bertopic_count)


with tab_products:
    st.subheader("Structured product catalog")
    products = load_products()
    if products is None:
        st.warning(f"`{PRODUCTS_PARQUET}` not found — run notebook 01 first.")
    else:
        st.write(
            f"**{products.height:,} cleaned products** across "
            f"**{products['source_category'].n_unique()} source categories**."
        )

        st.markdown("##### Catalog overview (from notebook 01)")
        show_image(PRODUCTS_DIR / "amazon_products_overview.png",
                   caption="Price · discount % · rating distributions")
        show_image(PRODUCTS_DIR / "amazon_subcategories.png",
                   caption="Top 20 sub-categories by product count")

        if "discount_pct" in products.columns:
            st.markdown("##### Top 25 deals by discount %")
            top_discount = (
                products.lazy()
                .filter((pl.col("discount_pct") > 0) & (pl.col("discount_pct") <= 95))
                .sort("discount_pct", descending=True)
                .head(25)
                .select(["name", "source_category", "actual_price",
                         "discount_price", "discount_pct"])
                .collect()
            )
            st.dataframe(top_discount.to_pandas(), use_container_width=True)

        if {"ratings", "no_of_ratings"}.issubset(products.columns):
            st.markdown("##### Top 25 products by Wilson-score lower bound (≥100 ratings)")
            # Wilson 95% lower bound on rating/5 — same formula used in notebook 01.
            z2 = 1.96 ** 2
            n  = pl.col("no_of_ratings")
            p  = pl.col("ratings") / 5.0
            wilson = (
                products.lazy()
                .filter((n >= 100) & pl.col("ratings").is_not_null())
                .with_columns(((p + z2 / (2 * n)) / (1 + z2 / n)).round(4).alias("wilson_score"))
                .sort("wilson_score", descending=True)
                .head(25)
                .select(["name", "source_category", "discount_price",
                         "ratings", "no_of_ratings", "wilson_score"])
                .collect()
            )
            st.dataframe(wilson.to_pandas(), use_container_width=True)


with tab_eda:
    st.subheader("Reviews — exploratory analysis")
    show_image(REVIEWS_DIR / "rating_distribution.png",
               caption="Rating distribution per category")
    show_image(REVIEWS_DIR / "review_length_dist.png",
               caption="Review text length distribution (log scale)")


with tab_sent:
    st.subheader("Monthly sentiment timeline")
    show_image(REVIEWS_DIR / "monthly_sentiment.png",
               caption="Monthly average sentiment per category")
    sentiment = load_sentiment()
    if sentiment is not None:
        st.markdown("##### Underlying monthly sentiment table")
        st.dataframe(
            sentiment.sort(["category", "dt"]).to_pandas(),
            use_container_width=True,
            height=400,
        )


with tab_embed:
    st.subheader("Review embedding space (UMAP 2D)")
    show_image(REVIEWS_DIR / "review_embedding_space.png",
               caption="UMAP projection — sentiment + category overlays")
    st.caption(
        "BERTopic cluster overlay is available interactively in the **Topics** "
        "tab (Document scatter) — the static PNG was dropped because 200+ topic "
        "labels squashed the scatter plot into the top sliver of the canvas."
    )


with tab_topics:
    st.subheader("BERTopic interactive visualisations")
    st.caption("Each panel is the original plotly HTML written by notebook 02.")
    # Hierarchy + heatmap have long topic labels; give them extra vertical room.
    topic_panels = [
        ("Intertopic distance map",   "bertopic_intertopic.html", 650),
        ("Top words per topic",       "bertopic_barchart.html",   700),
        ("Topic similarity heatmap",  "bertopic_heatmap.html",    900),
        ("Topic hierarchy",           "bertopic_hierarchy.html",  900),
        ("Document scatter",          "bertopic_documents.html",  800),
    ]
    for title, fname, height in topic_panels:
        with st.expander(title, expanded=False):
            show_html(REVIEWS_DIR / fname, height=height)


with tab_datamap:
    st.subheader("Document datamap")
    st.caption(
        "Interactive scatter of all reviews with topic labels and hover tooltips. "
        "The 58 MB HTML is streamed via an iframe (Streamlit static-file serving) "
        "rather than inlined, so the page itself stays light."
    )
    show_html(
        REVIEWS_DIR / "bertopic_document_datamap.html",
        height=900,
        static_url="app/static/bertopic_document_datamap.html",
    )
