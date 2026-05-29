"""
Amazon datasets end-to-end pipeline — combined script.

Sourced from:
  - 01_structured_amazon_products.ipynb   (structured: Polars + cuDF on product CSVs)
  - 02_unstructured_amazon_reviews.ipynb  (unstructured: embeddings + Milvus + BERTopic)

All tunable knobs live in the "Parameters" block below; everything else flows
top-to-bottom. Run with:

    python amazon_pipeline.py

For the cuDF GPU acceleration on pandas / sklearn, run via the cudf.pandas
launcher (preferred over the inline install when available):

    python -m cudf.pandas amazon_pipeline.py

Prerequisites:
  - RAPIDS env (cudf-polars, cuml, sentence_transformers, bertopic, hdbscan, umap-learn)
  - Kaggle CLI configured (~/.kaggle/kaggle.json) for the Products download
  - Network access for the Reviews 2023 download from mcauleylab.ucsd.edu
"""


# =============================================================================
# Imports
# =============================================================================
import copy
import gzip
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# =============================================================================
# Parameters — edit these to control the pipeline
# =============================================================================

# ── Shared ───────────────────────────────────────────────────────────────────
GPU_DEVICE = 0

# ── 01: Amazon Products (structured) ─────────────────────────────────────────
PRODUCTS_DIR      = Path("./data/amazon").resolve()
PRODUCTS_DOWNLOAD = True                                  # set False to skip Kaggle download
SKIP_FILES        = {"Amazon-Products.csv", "Stores.csv"} # excluded from auto-discovery
COMBINED_CSV      = PRODUCTS_DIR / "Amazon-Products.csv"  # fallback if no per-category files

# ── 02: Amazon Reviews (unstructured) ────────────────────────────────────────
REVIEWS_DIR       = Path("./data/amazon_reviews").resolve()
REVIEWS_DOWNLOAD  = True
REVIEW_URLS = [
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Appliances.jsonl.gz",
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz",
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz",
]
REVIEW_FILES = {
    "Appliances":              REVIEWS_DIR / "Appliances.jsonl.gz",
    "Electronics":             REVIEWS_DIR / "Electronics.jsonl.gz",
    "Clothing_Shoes_Jewelry":  REVIEWS_DIR / "Clothing_Shoes_and_Jewelry.jsonl.gz",
}
MAX_REVIEWS_PER_CAT = 100_000   # cap per category — full files can be millions of rows
MIN_REVIEW_CHARS    = 50        # drop reviews shorter than this (low-signal)
VERIFIED_ONLY       = False     # True = keep only verified purchases

# Embedding model + vector DB
EMBEDDING_MODEL  = "nvidia/llama-3.2-nv-embedqa-1b-v2"
EMBED_DIM        = 2048   # nvidia/llama-3.2-nv-embedqa-1b-v2 outputs 2048-d
INFERENCE_BATCH  = 32
INGEST_BATCH     = 500
COLLECTION       = "amazon_reviews"
VECTORDB_BACKEND = "milvus"   # "milvus" or "elastic"
MILVUS_HOST      = "localhost"
MILVUS_PORT      = 19530
# Milvus Lite (embedded, no server) — set to None or "http://host:port" for a real server.
MILVUS_URI       = str(REVIEWS_DIR / "milvus_reviews.db")
MILVUS_RESET     = True   # clear Milvus Lite db on startup
ELASTIC_HOST     = "localhost"
ELASTIC_PORT     = 9200

# Topic modeling viz
TOP_N_TOPICS_VIZ = 30   # cap on legend / scatter labels (BERTopic can produce 200+ topics)

# ── 10: Optional Presto fallback (from 01 Step 10) ───────────────────────────
USE_PRESTO_FALLBACK = False   # True = also run the SQL/Presto path against the cleaned parquet
PRESTO_HOST_PORT    = 8089
PRESTO_CONTAINER    = "amazon-presto"


# =============================================================================
# Setup helpers (GPU detection, Polars collect wrappers)
# =============================================================================
PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

try:
    pl.LazyFrame({"x": [1]}).collect(engine=pl.GPUEngine(device=GPU_DEVICE, raise_on_fail=True))
    GPU_ENGINE    = pl.GPUEngine(device=GPU_DEVICE, raise_on_fail=False)
    GPU_AVAILABLE = True
except Exception as ex:
    GPU_ENGINE    = None
    GPU_AVAILABLE = False
    print(f"Polars GPU not available ({ex}) — CPU fallback active")

# Torch device for embedding model
try:
    import torch
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
    print("PyTorch not installed — embeddings will fail until it is")


def _engine(use_gpu: bool):
    return GPU_ENGINE if use_gpu and GPU_ENGINE is not None else "cpu"


def collect(lf: pl.LazyFrame, use_gpu: bool = GPU_AVAILABLE) -> pl.DataFrame:
    """Single-plan collect on the shared GPU engine (silent op-level CPU fallback)."""
    return lf.collect(engine=_engine(use_gpu))


def collect_all(lfs: list[pl.LazyFrame], use_gpu: bool = GPU_AVAILABLE) -> list[pl.DataFrame]:
    """Fuse N lazy plans into one optimizer pass — shared scans/filters run once."""
    engine = _engine(use_gpu)
    try:
        return pl.collect_all(lfs, engine=engine)
    except TypeError:
        return pl.collect_all(lfs, engine="gpu" if use_gpu and GPU_ENGINE is not None else "cpu")


# =============================================================================
# 01 — Amazon Products structured pipeline
# =============================================================================

# ── Step 0: optional Kaggle download ─────────────────────────────────────────
if PRODUCTS_DOWNLOAD:
    subprocess.run(
        ["kaggle", "datasets", "download",
         "-d", "lokeshparab/amazon-products-dataset",
         "--unzip", "-p", str(PRODUCTS_DIR)],
        check=False,
    )

# ── Step 1: discover and load per-category CSVs ──────────────────────────────
READ_OPTS = dict(
    infer_schema_length=5000,
    ignore_errors=True,
    null_values=["", "NA", "N/A", "null", "None"],
)

category_csvs = sorted(
    p for p in PRODUCTS_DIR.glob("*.csv") if p.name not in SKIP_FILES
)

if category_csvs:
    frames = [
        pl.scan_csv(p, **READ_OPTS).with_columns(pl.lit(p.stem).alias("source_category"))
        for p in category_csvs
    ]
elif COMBINED_CSV.exists():
    lf  = pl.scan_csv(COMBINED_CSV, **READ_OPTS)
    src = pl.col("main_category") if "main_category" in lf.collect_schema().names() else pl.lit("All")
    frames = [lf.with_columns(src.alias("source_category"))]
else:
    raise FileNotFoundError(
        f"No CSV files found in {PRODUCTS_DIR}.\n"
        "Download from Kaggle:\n"
        f"  kaggle datasets download -d lokeshparab/amazon-products-dataset --unzip -p {PRODUCTS_DIR}"
    )

raw_lf     = pl.concat(frames, how="diagonal_relaxed")
raw_schema = raw_lf.collect_schema()
print(f"Discovered {len(frames)} CSV plan(s); {len(raw_schema)} columns: {raw_schema.names()}")


# ── Step 2: data cleaning ────────────────────────────────────────────────────
def clean_price(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name).cast(pl.Utf8)
        .str.replace_all(r"[₹$€£,\s]", "")
        .cast(pl.Float64, strict=False)
        .alias(col_name)
    )


def clean_rating(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name).cast(pl.Utf8)
        .str.extract(r"^([0-9]+\.?[0-9]*)", 1)
        .cast(pl.Float32, strict=False)
        .alias(col_name)
    )


def clean_rating_count(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name).cast(pl.Utf8)
        .str.replace_all(",", "")
        .cast(pl.Int64, strict=False)
        .alias(col_name)
    )


cols       = set(raw_schema.names())
price_cols = [c for c in ["discount_price", "actual_price"] if c in cols]
rating_col = "ratings"       if "ratings"       in cols else None
count_col  = "no_of_ratings" if "no_of_ratings" in cols else None

clean_exprs = [clean_price(c) for c in price_cols]
if rating_col:
    clean_exprs.append(clean_rating(rating_col))
if count_col:
    clean_exprs.append(clean_rating_count(count_col))

products_lf = raw_lf.with_columns(clean_exprs)

if "discount_price" in cols and "actual_price" in cols:
    products_lf = products_lf.with_columns([
        ((pl.col("actual_price") - pl.col("discount_price"))
         / pl.col("actual_price") * 100.0).round(2).alias("discount_pct"),
        (pl.col("actual_price") - pl.col("discount_price")).alias("savings"),
    ])

schema_cols = products_lf.collect_schema().names()
keep_cols = [c for c in ["name", "main_category", "sub_category", "source_category",
                          "ratings", "no_of_ratings", "discount_price", "actual_price",
                          "discount_pct", "savings"] if c in schema_cols]
products_lf = (
    products_lf
    .select(keep_cols)
    .drop_nulls(subset=[c for c in ["discount_price", "name"] if c in keep_cols])
)

products = collect(products_lf, use_gpu=GPU_AVAILABLE)


# ── Step 4: category-level aggregation ───────────────────────────────────────
agg_exprs = [pl.len().alias("product_count")]
if "discount_price" in products.columns:
    agg_exprs += [
        pl.col("discount_price").min().alias("price_min"),
        pl.col("discount_price").max().alias("price_max"),
        pl.col("discount_price").mean().round(2).alias("price_mean"),
        pl.col("discount_price").median().alias("price_median"),
    ]
if "discount_pct" in products.columns:
    agg_exprs.append(pl.col("discount_pct").mean().round(2).alias("avg_discount_pct"))
if "ratings" in products.columns:
    agg_exprs.append(pl.col("ratings").mean().round(3).alias("avg_rating"))
if "no_of_ratings" in products.columns:
    agg_exprs.append(pl.col("no_of_ratings").sum().alias("total_ratings"))

group_col = "sub_category" if "sub_category" in products.columns else "source_category"

category_stats = collect(
    products.lazy()
    .filter(pl.col(group_col).is_not_null())
    .group_by(group_col)
    .agg(agg_exprs)
    .sort("product_count", descending=True),
    use_gpu=GPU_AVAILABLE,
)


# ── Step 5: top-discount products per category ───────────────────────────────
if "discount_pct" in products.columns:
    top_deals = collect(
        products.lazy()
        .filter((pl.col("discount_pct") > 0) & (pl.col("discount_pct") <= 95))
        .sort("discount_pct", descending=True)
        .group_by("source_category")
        .head(5)
        .select([
            "source_category", "name",
            "actual_price", "discount_price", "discount_pct",
            *(["ratings"] if "ratings" in products.columns else []),
        ])
        .sort(["source_category", "discount_pct"], descending=[False, True]),
        use_gpu=GPU_AVAILABLE,
    )


# ── Step 7: Wilson confidence score on ratings ───────────────────────────────
if "ratings" in products.columns and "no_of_ratings" in products.columns:
    z2 = 1.96 ** 2
    n  = pl.col("no_of_ratings")
    p  = pl.col("ratings") / 5.0

    rated_lf = (
        products.lazy()
        .filter(n.is_not_null() & (n > 0) & pl.col("ratings").is_not_null())
        .with_columns(
            ((p + z2 / (2 * n)) / (1 + z2 / n)).round(4).alias("wilson_score")
        )
    )

    top_rated_lf = (
        rated_lf
        .filter(pl.col("no_of_ratings") >= 100)
        .sort("wilson_score", descending=True)
        .head(10)
        .select(["name", "source_category", "discount_price", "ratings",
                 "no_of_ratings", "wilson_score"])
    )

    rated, top_rated = collect_all([rated_lf, top_rated_lf], use_gpu=GPU_AVAILABLE)


# ── Step 8: overview plots ───────────────────────────────────────────────────
n_cats     = products["source_category"].n_unique()
fig_height = max(6, n_cats * 0.05)
fig, axes  = plt.subplots(1, 3, figsize=(20, fig_height))

has_disc = "discount_pct" in products.columns
has_rat  = "ratings"      in products.columns

prices_lf = (
    products.lazy()
    .filter(pl.col("discount_price").is_not_null() & (pl.col("discount_price") > 0))
    .select(["source_category", "discount_price"])
)

plans: list[pl.LazyFrame] = [prices_lf]
if has_disc:
    plans.append(
        products.lazy()
        .filter(
            pl.col("discount_pct").is_not_null()
            & (pl.col("discount_pct") >= 0)
            & (pl.col("discount_pct") <= 95)
        )
        .group_by("source_category")
        .agg(pl.col("discount_pct").mean().round(1).alias("avg_disc"))
        .sort("avg_disc", descending=True)
        .head(20)
    )
if has_rat:
    plans.append(
        products.lazy()
        .filter(pl.col("ratings").is_not_null())
        .select(["source_category", "ratings"])
    )

results   = iter(collect_all(plans, use_gpu=GPU_AVAILABLE))
prices_df = next(results)
cat_disc  = next(results) if has_disc else None
rat_df    = next(results) if has_rat  else None

prices_by_cat = {
    cat: df["discount_price"].to_numpy()
    for cat, df in prices_df.partition_by("source_category", as_dict=True).items()
}
ratings_by_cat = (
    {} if rat_df is None
    else {
        cat: df["ratings"].to_numpy()
        for cat, df in rat_df.partition_by("source_category", as_dict=True).items()
    }
)

ax = axes[0]
for cat in sorted(prices_by_cat):
    prices_cat = prices_by_cat[cat]
    if len(prices_cat) > 0:
        ax.hist(np.log10(np.clip(prices_cat, 1, None)), bins=40, alpha=0.5, label=cat)
ax.set_xlabel("log10(Discount Price)")
ax.set_ylabel("Product Count")
ax.set_title("Price Distribution (log scale)")
ax.grid(alpha=0.3)

ax = axes[1]
if cat_disc is not None:
    cats  = cat_disc["source_category"].to_list()
    discs = cat_disc["avg_disc"].to_list()
    bars  = ax.barh(cats, discs, color="steelblue", alpha=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_xlabel("Avg Discount %")
    ax.set_title("Average Discount by Category")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.3, axis="x")
else:
    ax.text(0.5, 0.5, "discount_pct unavailable", ha="center", va="center")

ax = axes[2]
if ratings_by_cat:
    for cat in sorted(ratings_by_cat):
        r = ratings_by_cat[cat]
        if len(r) > 0:
            ax.hist(r, bins=np.linspace(1, 5, 30), alpha=0.5, label=cat)
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Product Count")
    ax.set_title("Rating Distribution by Category")
    ax.set_xlim(1, 5)
    ax.grid(alpha=0.3)
else:
    ax.text(0.5, 0.5, "ratings unavailable", ha="center", va="center")

plt.tight_layout()
plt.savefig(PRODUCTS_DIR / "amazon_products_overview.png", dpi=130, bbox_inches="tight")
plt.show()

# Sub-category bar chart
if "sub_category" in products.columns:
    top_subs = collect(
        products.lazy()
        .filter(pl.col("sub_category").is_not_null())
        .group_by(["source_category", "sub_category"])
        .agg(
            pl.len().alias("count"),
            *([pl.col("discount_price").median().round(0).alias("median_price")]
              if "discount_price" in products.columns else []),
        )
        .sort("count", descending=True)
        .head(20),
        use_gpu=GPU_AVAILABLE,
    )

    labels = [
        f"[{row['source_category'][:3]}] {row['sub_category'][:35]}"
        for row in top_subs.iter_rows(named=True)
    ]
    counts = top_subs["count"].to_list()
    palette = plt.cm.Set2(np.linspace(0, 1, len(set(top_subs["source_category"].to_list()))))
    cat_color = {c: palette[i] for i, c in enumerate(sorted(set(top_subs["source_category"].to_list())))}
    colors = [cat_color[r] for r in top_subs["source_category"].to_list()]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(labels, counts, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_xlabel("Product Count")
    ax.set_title("Top 20 Sub-Categories by Product Count")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(PRODUCTS_DIR / "amazon_subcategories.png", dpi=130)
    plt.show()


# ── Step 9: export clean parquet ─────────────────────────────────────────────
PRODUCTS_PARQUET = PRODUCTS_DIR / "amazon_products_clean.parquet"
products.write_parquet(PRODUCTS_PARQUET)


# =============================================================================
# 02 — Amazon Reviews unstructured pipeline
# =============================================================================

# ── Step 0: optional download + cache reset ──────────────────────────────────
if REVIEWS_DOWNLOAD:
    for url in REVIEW_URLS:
        target = REVIEWS_DIR / Path(url).name
        if not target.exists():
            subprocess.run(
                ["wget", url, "--no-check-certificate", "-P", str(REVIEWS_DIR)],
                check=False,
            )

if MILVUS_RESET:
    shutil.rmtree(REVIEWS_DIR / "milvus_reviews.db", ignore_errors=True)


# ── Step 1: load + filter reviews from JSONL.gz ──────────────────────────────
def unix_ms_to_iso(ts_ms) -> str:
    try:
        return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return datetime.now(tz=timezone.utc).isoformat()


def stable_id(user_id: str, asin: str, idx: int) -> str:
    digest = hashlib.md5(f"{user_id}:{asin}:{idx}".encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def load_reviews_jsonl_gz(
    path: Path,
    category: str,
    max_rows: int,
    min_chars: int = MIN_REVIEW_CHARS,
    verified_only: bool = VERIFIED_ONLY,
) -> list[dict]:
    records: list[dict] = []
    seen = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen += 1
            text     = (rec.get("text") or "").strip()
            rating   = rec.get("rating")
            verified = rec.get("verified_purchase", True)
            if not text or len(text) < min_chars or rating is None:
                continue
            if verified_only and not verified:
                continue
            records.append({
                "category":     category,
                "id":           stable_id(rec.get("user_id", ""), rec.get("asin", ""), seen),
                "asin":         rec.get("parent_asin") or rec.get("asin", "UNKNOWN"),
                "title":        (rec.get("title") or "").strip()[:1024],
                "text":         text[:8192],
                "rating":       float(rating),
                "timestamp":    unix_ms_to_iso(rec.get("timestamp", 0)),
                "helpful_vote": int(rec.get("helpful_vote") or 0),
                "verified":     bool(verified),
                "text_len":     len(text[:8192]),
            })
            if len(records) >= max_rows:
                break
    return records


all_records: list[dict] = []
for category, path in REVIEW_FILES.items():
    if not path.exists():
        continue
    all_records.extend(load_reviews_jsonl_gz(path, category, max_rows=MAX_REVIEWS_PER_CAT))

if not all_records:
    raise RuntimeError(
        "No reviews loaded. Download the datasets from:\n"
        "   https://mcauleylab.ucsd.edu\n"
        "Then place the .jsonl.gz files in REVIEWS_DIR and re-run."
    )

df_reviews = collect(pl.from_dicts(all_records).lazy())


# ── Step 2: EDA plots ────────────────────────────────────────────────────────
eda = collect(df_reviews.lazy().select(["category", "rating", "text_len"]))
by_cat = eda.partition_by("category", as_dict=True)
cats = sorted(by_cat.keys())

fig, axes = plt.subplots(1, len(cats), figsize=(6 * len(cats), 6), sharey=True)
if len(cats) == 1:
    axes = [axes]
for ax, cat in zip(axes, cats):
    ratings = by_cat[cat]["rating"].to_list()
    star_counts = Counter(int(r) for r in ratings)
    stars  = [1, 2, 3, 4, 5]
    counts = [star_counts.get(s, 0) for s in stars]
    colors = ["#d32f2f", "#ef6c00", "#f9a825", "#7cb342", "#2e7d32"]
    ax.bar([str(s) + "★" for s in stars], counts, color=colors, alpha=0.85)
    ax.set_title(f"{cat}\n({len(ratings):,} reviews)")
    ax.set_xlabel("Star Rating")
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("Review Count")
plt.tight_layout()
plt.savefig(REVIEWS_DIR / "rating_distribution.png", dpi=120, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(10, 4))
for cat in cats:
    lengths = by_cat[cat]["text_len"].to_numpy()
    ax.hist(np.log10(np.clip(lengths, 1, None)), bins=50, alpha=0.5, label=cat)
ax.set_xlabel("lg(Review Length in chars)")
ax.set_ylabel("Count")
ax.set_title("Review Text Length Distribution")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REVIEWS_DIR / "review_length_dist.png", dpi=120)
plt.show()


# ── Step 3: pipeline schema ──────────────────────────────────────────────────
@dataclass
class AmazonReview:
    """Pipeline record for a single Amazon customer review."""
    id:              str
    title:           str
    body:            str
    asin:            str
    category:        str
    rating:          float
    sentiment_score: float  # (rating − 3) / 2  →  [-1.0, +1.0]
    timestamp:       str


def rating_to_sentiment(rating: float) -> float:
    return round(float(np.clip((rating - 3.0) / 2.0, -1.0, 1.0)), 4)


reviews: list[AmazonReview] = [
    AmazonReview(
        id=rec["id"],
        title=rec["title"] or rec["text"][:120],
        body=rec["text"],
        asin=rec["asin"],
        category=rec["category"],
        rating=rec["rating"],
        sentiment_score=rating_to_sentiment(rec["rating"]),
        timestamp=rec["timestamp"],
    )
    for rec in all_records
]
review_texts: list[str] = [f"{r.title}. {r.body}".strip(" .") for r in reviews]
sentiments = np.array([r.sentiment_score for r in reviews])


# ── Step 4: embeddings ───────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer


def embed_texts(texts: list[str], prompt: str = "passage") -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    return model.encode(
        texts,
        prompt=prompt,
        batch_size=INFERENCE_BATCH,
        device=DEVICE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


embeddings = embed_texts(review_texts)


# ── Step 5: vector DB setup ──────────────────────────────────────────────────
def setup_vector_store(
    backend: str,
    *,
    collection: str,
    dim: int,
    milvus_uri: str | None = None,
    milvus_host: str = "localhost",
    milvus_port: int = 19530,
    elastic_host: str = "localhost",
    elastic_port: int = 9200,
):
    if backend == "milvus":
        from pymilvus import MilvusClient, DataType

        uri    = milvus_uri or f"http://{milvus_host}:{milvus_port}"
        client = MilvusClient(uri=uri)
        if client.has_collection(collection):
            client.drop_collection(collection)

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id",              DataType.VARCHAR,      is_primary=True, max_length=128)
        schema.add_field("embedding",       DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("title",           DataType.VARCHAR,      max_length=2048)
        schema.add_field("body",            DataType.VARCHAR,      max_length=16384)
        schema.add_field("asin",            DataType.VARCHAR,      max_length=64)
        schema.add_field("category",        DataType.VARCHAR,      max_length=64)
        schema.add_field("rating",          DataType.FLOAT)
        schema.add_field("sentiment_score", DataType.FLOAT)
        schema.add_field("timestamp",       DataType.VARCHAR,      max_length=32)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(collection, schema=schema, index_params=index_params)
        return client

    if backend == "elastic":
        from elasticsearch import Elasticsearch
        url    = f"http://{elastic_host}:{elastic_port}"
        client = Elasticsearch(url, request_timeout=60)
        if not client.ping():
            raise ConnectionError(f"Elasticsearch did not respond at {url}")
        if client.indices.exists(index=collection):
            client.indices.delete(index=collection)
        mapping = {
            "mappings": {
                "properties": {
                    "embedding":       {"type": "dense_vector", "dims": dim,
                                        "index": True, "similarity": "cosine"},
                    "title":           {"type": "text"},
                    "body":            {"type": "text"},
                    "asin":            {"type": "keyword"},
                    "category":        {"type": "keyword"},
                    "rating":          {"type": "float"},
                    "sentiment_score": {"type": "float"},
                    "timestamp":       {"type": "date"},
                }
            }
        }
        client.indices.create(index=collection, body=mapping)
        return client

    raise ValueError(f"Unknown VECTORDB_BACKEND: {backend!r} (use 'milvus' or 'elastic')")


try:
    vdb_client = setup_vector_store(
        VECTORDB_BACKEND,
        collection=COLLECTION,
        dim=int(embeddings.shape[1]),
        milvus_uri=MILVUS_URI,
        milvus_host=MILVUS_HOST,   milvus_port=MILVUS_PORT,
        elastic_host=ELASTIC_HOST, elastic_port=ELASTIC_PORT,
    )
    VDB_CONNECTED = True
except Exception as e:
    print(f"VectorDB not available: {e}")
    vdb_client    = None
    VDB_CONNECTED = False


# ── Step 6: ingest ───────────────────────────────────────────────────────────
def ingest_reviews(
    client, backend: str, *, collection: str, reviews: list, embeddings, batch_size: int,
) -> tuple[int, int]:
    assert len(reviews) == len(embeddings), "reviews / embeddings length mismatch"
    total = 0
    if backend == "milvus":
        for i in range(0, len(reviews), batch_size):
            batch_reviews = reviews[i : i + batch_size]
            batch_vecs    = embeddings[i : i + batch_size]
            rows = [
                {
                    "id":              r.id,
                    "embedding":       v.tolist(),
                    "title":           r.title[:2048],
                    "body":            r.body[:16384],
                    "asin":            r.asin,
                    "category":        r.category,
                    "rating":          float(r.rating),
                    "sentiment_score": float(r.sentiment_score),
                    "timestamp":       r.timestamp,
                }
                for r, v in zip(batch_reviews, batch_vecs)
            ]
            client.insert(collection, rows)
            total += len(rows)
        client.flush(collection)
        stats = client.get_collection_stats(collection)
        count = int(stats.get("row_count", total))
    elif backend == "elastic":
        from elasticsearch.helpers import bulk

        def _actions():
            for r, v in zip(reviews, embeddings):
                yield {
                    "_index": collection,
                    "_id":    r.id,
                    "_source": {
                        "embedding":       v.tolist(),
                        "title":           r.title,
                        "body":            r.body,
                        "asin":            r.asin,
                        "category":        r.category,
                        "rating":          float(r.rating),
                        "sentiment_score": float(r.sentiment_score),
                        "timestamp":       r.timestamp,
                    },
                }

        success, _errors = bulk(client, _actions(), chunk_size=batch_size,
                                request_timeout=120, raise_on_error=False)
        total = success
        client.indices.refresh(index=collection)
        count = client.count(index=collection)["count"]
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
    return total, count


if VDB_CONNECTED:
    written, in_collection = ingest_reviews(
        vdb_client, VECTORDB_BACKEND,
        collection=COLLECTION,
        reviews=reviews,
        embeddings=embeddings,
        batch_size=INGEST_BATCH,
    )


# ── Step 7: semantic search ──────────────────────────────────────────────────
def semantic_search_vdb(query: str, *, top_k: int = 10) -> list:
    q_vec = embed_texts([query], prompt="query")[0].tolist()
    if VECTORDB_BACKEND == "milvus":
        hits = vdb_client.search(
            collection_name=COLLECTION,
            data=[q_vec],
            limit=top_k,
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["title", "category", "sentiment_score", "rating"],
        )[0]
        return [(h["entity"], h["distance"]) for h in hits]
    if VECTORDB_BACKEND == "elastic":
        resp = vdb_client.search(
            index=COLLECTION,
            knn={
                "field":          "embedding",
                "query_vector":   q_vec,
                "k":              top_k,
                "num_candidates": max(top_k * 10, 50),
            },
            source=["title", "category", "sentiment_score", "rating"],
        )
        return [(h["_source"], h["_score"]) for h in resp["hits"]["hits"]]
    raise ValueError(f"Unknown VECTORDB_BACKEND: {VECTORDB_BACKEND!r}")


def semantic_search_numpy(query: str, *, top_k: int = 10) -> list:
    q_vec   = embed_texts([query], prompt="query")[0]
    scores  = embeddings @ q_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(reviews[i], scores[i]) for i in top_idx]


search_fn = semantic_search_vdb if VDB_CONNECTED else semantic_search_numpy

QUERIES = [
    "stopped working after a few months, terrible quality",
    "battery drains too fast, disappointing",
    "easy to set up, works exactly as described",
    "great value for the price, highly recommend",
    "shipping was damaged and packaging broken",
    "too small, sizing runs very small",
    "noisy compressor keeps me awake at night",
    "excellent customer service resolved my issue quickly",
]
for query in QUERIES:
    search_fn(query, top_k=5)

for cat, query in [
    ("Appliances",             "energy consumption too high, electricity bill increased"),
    ("Electronics",            "screen flickering and display issues after update"),
    ("Clothing_Shoes_Jewelry", "material feels cheap and itchy, not as pictured"),
]:
    if any(r.category == cat for r in reviews):
        search_fn(query, top_k=5)


# ── Step 8: monthly sentiment timeseries ─────────────────────────────────────
df_ts_lf = (
    df_reviews.lazy()
    .with_columns(
        ((pl.col("rating") - 3.0) / 2.0).clip(-1.0, 1.0).alias("sentiment"),
        pl.col("timestamp").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%z", strict=False).alias("dt"),
    )
    .filter(pl.col("dt").is_not_null())
)
monthly_lf = (
    df_ts_lf.sort("dt")
    .group_by_dynamic("dt", every="1mo", group_by="category")
    .agg([
        pl.col("sentiment").mean().round(3).alias("avg_sentiment"),
        pl.col("sentiment").count().alias("review_count"),
    ])
    .sort(["category", "dt"])
)
df_ts, monthly = collect_all([df_ts_lf, monthly_lf])

# Monthly sentiment plot
cats_available = sorted(collect(monthly.lazy().select("category").unique())["category"].to_list())
slices = collect_all([
    monthly.lazy().filter(pl.col("category") == cat).sort("dt")
    for cat in cats_available
])
slice_by_cat = dict(zip(cats_available, slices))

fig, axes = plt.subplots(len(cats_available), 1,
                         figsize=(13, 4 * len(cats_available)), sharex=False)
if len(cats_available) == 1:
    axes = [axes]
for ax, cat in zip(axes, cats_available):
    cat_data = slice_by_cat[cat]
    dts    = cat_data["dt"].to_list()
    sents  = cat_data["avg_sentiment"].to_list()
    counts = cat_data["review_count"].to_list()
    ax2 = ax.twinx()
    ax2.bar(dts, counts, color="steelblue", alpha=0.25, label="Review count")
    ax2.set_ylabel("Reviews / month", color="steelblue", fontsize=9)
    ax.plot(dts, sents, color="black", linewidth=1.5, zorder=3)
    ax.fill_between(dts, 0, sents, where=[s >= 0 for s in sents],
                    color="green", alpha=0.3, zorder=2)
    ax.fill_between(dts, 0, sents, where=[s < 0 for s in sents],
                    color="red", alpha=0.3, zorder=2)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Avg Sentiment", fontsize=9)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"{cat} — Monthly Review Sentiment")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", rotation=90)
    ax.grid(alpha=0.2)
plt.xticks(rotation=120)
plt.tight_layout()
plt.savefig(REVIEWS_DIR / "monthly_sentiment.png", dpi=120)
plt.show()

# Export monthly sentiment for the Streamlit dashboard
SENTIMENT_PARQUET = REVIEWS_DIR / "amazon_reviews_sentiment.parquet"
monthly.write_parquet(SENTIMENT_PARQUET)


# ── Step 9: BERTopic clustering ──────────────────────────────────────────────
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

umap_model    = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
hdbscan_model = HDBSCAN(min_cluster_size=50, min_samples=10,
                        gen_min_span_tree=True, prediction_data=True)
topic_model   = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)
topics, probs = topic_model.fit_transform(review_texts, embeddings=embeddings)

# 2D UMAP for visualization (separate from the 5D one used by BERTopic)
umap_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.05,
               metric="cosine", random_state=42)
coords = umap_2d.fit_transform(embeddings)


# ── Step 10: embedding-space visualizations ──────────────────────────────────
sentiments_all = np.array([r.sentiment_score for r in reviews])
categories_all = [r.category for r in reviews]
topics_all     = np.asarray(topics)

x_lo, x_hi = np.percentile(coords[:, 0], [1, 99])
y_lo, y_hi = np.percentile(coords[:, 1], [1, 99])
x_pad = (x_hi - x_lo) * 0.03
y_pad = (y_hi - y_lo) * 0.03
xlim = (x_lo - x_pad, x_hi + x_pad)
ylim = (y_lo - y_pad, y_hi + y_pad)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

# Left — by sentiment
sc = ax1.scatter(coords[:, 0], coords[:, 1],
                 c=sentiments_all, cmap="RdYlGn",
                 vmin=-1.0, vmax=1.0, s=7, alpha=0.65)
cbar = fig.colorbar(sc, ax=ax1, pad=0.01)
cbar.set_label("Sentiment (from star rating)")
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.set_ticklabels(["1★", "2★", "3★", "4★", "5★"])
ax1.set_title("Coloured by Customer Sentiment (Star Rating)")
ax1.set_xlabel("UMAP-1")
ax1.set_ylabel("UMAP-2")
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.grid(alpha=0.15)

# Middle — by category
cat_unique    = sorted(set(categories_all))
cat_palette   = cm.get_cmap("Set1", len(cat_unique))
cat_color_map = {c: cat_palette(i) for i, c in enumerate(cat_unique)}
cat_colors    = [cat_color_map[c] for c in categories_all]

ax2.scatter(coords[:, 0], coords[:, 1], c=cat_colors, s=7, alpha=0.55)
for cat, color in cat_color_map.items():
    ax2.scatter([], [], c=[color], label=cat, s=30)
ax2.set_title("Coloured by Product Category")
ax2.set_xlabel("UMAP-1")
ax2.set_ylabel("UMAP-2")
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(alpha=0.15)

# Right — by BERTopic cluster
noise_mask   = topics_all == -1
cluster_mask = ~noise_mask
n_topics     = int(topics_all.max()) + 1 if cluster_mask.any() else 0
topic_cmap   = cm.get_cmap("tab20", max(n_topics, 1))

ax3.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
            c="lightgrey", s=5, alpha=0.2, linewidths=0)
ax3.scatter(coords[cluster_mask, 0], coords[cluster_mask, 1],
            c=topics_all[cluster_mask], cmap=topic_cmap,
            s=7, alpha=0.65, linewidths=0)
ax3.set_title(f"Coloured by BERTopic Cluster ({n_topics} topics)")
ax3.set_xlabel("UMAP-1")
ax3.set_ylabel("UMAP-2")
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.grid(alpha=0.15)

plt.suptitle("Amazon Reviews 2023 — Embedding Space (UMAP 2D)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(REVIEWS_DIR / "review_embedding_space.png", dpi=130, bbox_inches="tight")
plt.show()


# ── Step 11: BERTopic HTML visualizations ────────────────────────────────────
# fig = topic_model.visualize_topics()
# fig.write_html(str(REVIEWS_DIR / "bertopic_intertopic.html"))
#
# fig = topic_model.visualize_barchart(top_n_topics=12, n_words=8, height=300)
# fig.write_html(str(REVIEWS_DIR / "bertopic_barchart.html"))
#
# # Top-60 heatmap with explicit margins for rotated labels
# fig = topic_model.visualize_heatmap(top_n_topics=60, width=1100, height=1100)
# fig.update_xaxes(tickangle=-90)
# fig.update_layout(margin=dict(l=180, r=140, b=240, t=120))
# fig.update_traces(
#     selector=dict(type="heatmap"),
#     colorbar=dict(
#         title=dict(text="Similarity Score", side="top"),
#         len=0.85, y=0.5, yanchor="middle",
#     ),
# )
# fig.write_html(str(REVIEWS_DIR / "bertopic_heatmap.html"))
#
# # Top-60 hierarchy dendrogram
# fig = topic_model.visualize_hierarchy(top_n_topics=60, width=1100, height=1200)
# fig.update_layout(margin=dict(l=300, r=20, b=40, t=80))
# fig.write_html(str(REVIEWS_DIR / "bertopic_hierarchy.html"))
#
# # Top-N documents scatter (reuses the 2D coords from above)
# top_topics = (
#     topic_model.get_topic_info()
#     .query("Topic != -1")
#     .nlargest(TOP_N_TOPICS_VIZ, "Count")["Topic"]
#     .tolist()
# )
# fig = topic_model.visualize_documents(
#     review_texts,
#     embeddings=embeddings,
#     reduced_embeddings=coords,
#     topics=top_topics,
#     hide_annotations=True,
#     sample=min(1.0, 8000 / max(len(review_texts), 1)),
#     width=1600,
# )
# x_lo, x_hi = np.percentile(coords[:, 0], [1, 99])
# y_lo, y_hi = np.percentile(coords[:, 1], [1, 99])
# x_pad = (x_hi - x_lo) * 0.03
# y_pad = (y_hi - y_lo) * 0.03
# fig.update_layout(
#     xaxis=dict(range=[x_lo - x_pad, x_hi + x_pad]),
#     yaxis=dict(range=[y_lo - y_pad, y_hi + y_pad]),
#     margin=dict(l=60, r=500, b=60, t=80),
#     legend=dict(font=dict(size=10), itemsizing="constant",
#                 x=1.02, xanchor="left", y=1.0),
# )
# fig.write_html(str(REVIEWS_DIR / "bertopic_documents.html"))

# DataMap on the full corpus.
# Kept active: this file is loaded by the Streamlit dashboard (app.py) as the
# "Topic Modeling - Document Datamap" panel.
# NOTE: the original code subsampled to 1% because of a datamapplot 256-color
# palette bug. We try the full corpus; if it breaks, revert to a sample.
fig = topic_model.visualize_document_datamap(
    review_texts,
    embeddings=embeddings,
    interactive=True,
    int_datamap_kwds={"darkmode": True},
)
fig.save(str(REVIEWS_DIR / "bertopic_document_datamap.html"))


# ── Step 12: teardown vector DB connection ──────────────────────────────────
if VDB_CONNECTED and vdb_client is not None:
    if VECTORDB_BACKEND == "milvus":
        vdb_client.close()
    elif VECTORDB_BACKEND == "elastic":
        vdb_client.close()


# =============================================================================
# 10 — Optional Presto / SQL fallback (from 01 Step 10)
# =============================================================================
# Runs only when USE_PRESTO_FALLBACK is True. Spins up a single-node Presto
# container, registers the cleaned products parquet from above as a Hive
# external table, and re-runs the Step 4/5 aggregations in SQL. Output
# overwrites `category_stats` and `top_deals` for parity with the Polars path.
#
# Requires: docker, `pip install presto-python-client`.

if USE_PRESTO_FALLBACK:
    PRESTO_CATALOG   = Path("./presto/catalog").resolve()
    PRESTO_TABLE_DIR = PRODUCTS_DIR / "presto_products"
    PRESTO_CATALOG.mkdir(parents=True, exist_ok=True)
    PRESTO_TABLE_DIR.mkdir(exist_ok=True)

    # Stage the parquet in a Hive-shaped directory.
    shutil.copy2(PRODUCTS_PARQUET, PRESTO_TABLE_DIR / "products.parquet")

    # Only override Presto's hive catalog config — keep all other defaults.
    (PRESTO_CATALOG / "amazon.properties").write_text(
        "connector.name=hive-hadoop2\n"
        "hive.metastore=file\n"
        "hive.metastore.catalog.dir=file:/var/lib/presto/data/hive/metastore\n"
        "hive.allow-drop-table=true\n"
    )

    # Drop any prior container, then start fresh.
    subprocess.run(["docker", "rm", "-f", PRESTO_CONTAINER], capture_output=True)
    subprocess.run(
        ["docker", "run", "-d",
         "--name", PRESTO_CONTAINER,
         "-p", f"{PRESTO_HOST_PORT}:8080",
         "-v", f"{(PRESTO_CATALOG / 'amazon.properties').resolve()}"
               ":/opt/presto-server/etc/catalog/amazon.properties:ro",
         "-v", f"{PRESTO_TABLE_DIR.resolve()}:/data/products:ro",
         "prestodb/presto:latest"],
        check=True,
    )

    # Poll /v1/info until Presto is ready.
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{PRESTO_HOST_PORT}/v1/info", timeout=2,
            ) as resp:
                info = json.loads(resp.read())
                if info.get("starting") is False:
                    break
        except Exception:
            pass
        time.sleep(2)
    else:
        raise RuntimeError(
            f"Presto did not start in 180s — check `docker logs {PRESTO_CONTAINER}`"
        )

    from prestodb import dbapi as prestodb_dbapi  # pip install presto-python-client

    conn = prestodb_dbapi.connect(
        host="localhost", port=PRESTO_HOST_PORT,
        user="amazon-pipeline",
        catalog="amazon", schema="default",
    )

    def _sql(query: str):
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return rows, cols

    def _sql_df(query: str) -> pl.DataFrame:
        rows, cols = _sql(query)
        return pl.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pl.DataFrame()

    # File-based Hive metastore rejects schema-level locations; the table's
    # `external_location` is what actually points Hive at the parquet.
    _sql("CREATE SCHEMA IF NOT EXISTS amazon.default")
    _sql("""
        CREATE TABLE IF NOT EXISTS amazon.default.products (
            name            VARCHAR,
            main_category   VARCHAR,
            sub_category    VARCHAR,
            source_category VARCHAR,
            ratings         REAL,
            no_of_ratings   BIGINT,
            discount_price  DOUBLE,
            actual_price    DOUBLE,
            discount_pct    DOUBLE,
            savings         DOUBLE
        )
        WITH (
            external_location = 'file:///data/products',
            format = 'PARQUET'
        )
    """)

    # Step 4 mirror — category aggregation in SQL.
    category_stats = _sql_df("""
        SELECT
            sub_category,
            COUNT(*)                                AS product_count,
            MIN(discount_price)                     AS price_min,
            MAX(discount_price)                     AS price_max,
            ROUND(AVG(discount_price), 2)           AS price_mean,
            APPROX_PERCENTILE(discount_price, 0.5)  AS price_median,
            ROUND(AVG(discount_pct), 2)             AS avg_discount_pct,
            ROUND(AVG(ratings), 3)                  AS avg_rating,
            SUM(no_of_ratings)                      AS total_ratings
        FROM amazon.default.products
        WHERE sub_category IS NOT NULL
        GROUP BY sub_category
        ORDER BY product_count DESC
    """)

    # Step 5 mirror — top deals per category via window-rank.
    top_deals = _sql_df("""
        WITH ranked AS (
            SELECT
                source_category,
                name,
                actual_price,
                discount_price,
                discount_pct,
                ratings,
                ROW_NUMBER() OVER (
                    PARTITION BY source_category
                    ORDER BY discount_pct DESC
                ) AS rn
            FROM amazon.default.products
            WHERE discount_pct > 0 AND discount_pct <= 95
        )
        SELECT source_category, name, actual_price, discount_price, discount_pct, ratings
        FROM ranked
        WHERE rn <= 5
        ORDER BY source_category, discount_pct DESC
    """)

    conn.close()
    subprocess.run(["docker", "rm", "-f", PRESTO_CONTAINER], capture_output=True)
