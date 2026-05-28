"""Streamlit dashboard for the Amazon products + reviews pipelines.

Run with:
    ./run.sh                 # executes notebooks 01 & 02, then launches streamlit
    streamlit run app.py     # dashboard only (assumes artifacts already on disk)
"""
import subprocess
import sys
from pathlib import Path

import polars as pl
import streamlit as st

ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data"
PRODUCTS_DIR  = DATA_DIR / "amazon"
REVIEWS_DIR   = DATA_DIR / "amazon_reviews"

PRODUCTS_PARQUET  = PRODUCTS_DIR / "amazon_products_clean.parquet"
SENTIMENT_PARQUET = REVIEWS_DIR  / "amazon_reviews_sentiment.parquet"

st.set_page_config(
    page_title="Amazon Data Pipelines",
    layout="wide",
    initial_sidebar_state="expanded",
)

PIPELINE_SCRIPT = ROOT / "amazon_pipeline.py"

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
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Missing artifact `{path.name}` — re-run the notebook to generate it.")


def show_html(path: Path, height: int = 700) -> None:
    """Embed a saved plotly HTML with scrollbars when content overflows."""
    if not path.exists():
        st.info(f"Missing artifact `{path.name}` — re-run the notebook to generate it.")
        return
    html = path.read_text()
    # Plotly's saved HTML clips multi-digit legend labels because the default
    # legend.itemwidth (30px) only fits ~1 character at this font size. Patch
    # the figure after it mounts so two/three-digit topic IDs render in full.
    patch = """
<style>.legend .scrollbox{overflow:visible!important}</style>
<script>
(function () {
  const widenLegend = () => {
    const plot = document.querySelector('.js-plotly-plot');
    if (plot && window.Plotly) {
      window.Plotly.relayout(plot, {'legend.itemwidth': 80});
      return true;
    }
    return false;
  };
  if (!widenLegend()) {
    const timer = setInterval(() => { if (widenLegend()) clearInterval(timer); }, 150);
    setTimeout(() => clearInterval(timer), 5000);
  }
})();
</script>
"""
    html = html.replace("</body>", patch + "</body>", 1)
    st.components.v1.html(html, height=height, scrolling=True)


# ── Sidebar: pipeline control ─────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Pipeline control")
    st.caption(
        "Runs `amazon_pipeline.py` end-to-end with GPU acceleration "
        "(via `python -m cudf.pandas -m cuml.accel`). Takes several minutes — "
        "the page blocks until it finishes, then auto-refreshes."
    )
    if not PIPELINE_SCRIPT.exists():
        st.error(f"`{PIPELINE_SCRIPT.name}` not found next to app.py.")
    elif st.button("▶ Run pipeline", type="primary", use_container_width=True):
        cmd = [
            sys.executable, "-m", "cudf.pandas", "-m", "cuml.accel",
            str(PIPELINE_SCRIPT),
        ]
        with st.status("Running pipeline — this can take ~10 min", expanded=True) as status:
            log_box = st.empty()
            log_lines: list[str] = []
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                # Keep only the last 30 lines visible — the full stream is huge
                # and would lag the browser if we appended every model-load row.
                log_box.code("\n".join(log_lines[-30:]), language="text")
            rc = proc.wait()
            if rc == 0:
                status.update(label="Pipeline complete — reloading artifacts",
                              state="complete", expanded=False)
                st.cache_data.clear()
                st.rerun()
            else:
                status.update(label=f"Pipeline failed (exit code {rc})",
                              state="error")


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
| `02_unstructured_amazon_reviews.ipynb` | McAuley Lab Amazon Reviews 2023 | Embedding (Nemotron 1B), Milvus vector store, BERTopic clustering | `sentiment parquet` + topic HTMLs |

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
            st.dataframe(top_discount.to_pandas(), width="stretch")

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
            st.dataframe(wilson.to_pandas(), width="stretch")


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
            width="stretch",
            height=400,
        )


with tab_embed:
    st.subheader("Review embedding space (UMAP 2D)")
    show_image(REVIEWS_DIR / "review_embedding_space.png",
               caption="UMAP projection — sentiment + category + BERTopic cluster overlays")
    st.caption(
        "Topic panel uses colour only (no legend) because 200+ HDBSCAN clusters "
        "would otherwise dominate the canvas. Hover the **Document scatter** in "
        "the Topics tab for per-cluster keyword labels."
    )


with tab_topics:
    st.subheader("BERTopic interactive visualisations")
    st.caption("Each panel is the original plotly HTML written by notebook 02.")
    # Hierarchy + heatmap have long topic labels; give them extra vertical room.
    topic_panels = [
        ("Intertopic distance map",   "bertopic_intertopic.html", 650),
        ("Top words per topic",       "bertopic_barchart.html",   700),
        ("Topic similarity heatmap",  "bertopic_heatmap.html",    1250),
        ("Topic hierarchy",           "bertopic_hierarchy.html",  1050),
        ("Document scatter",          "bertopic_documents.html",  1000),
    ]
    for title, fname, height in topic_panels:
        with st.expander(title, expanded=False):
            show_html(REVIEWS_DIR / fname, height=height)


with tab_datamap:
    st.subheader("Document datamap")
    st.caption(
        "Interactive scatter of all reviews with topic labels and hover tooltips."
    )
    show_html(
        REVIEWS_DIR / "bertopic_document_datamap.html",
        height=900,
    )
