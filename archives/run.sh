#!/usr/bin/env bash
# Execute the two pipeline notebooks, then launch the Streamlit dashboard.
# nbconvert runs them in-process (rapids env's "python3" kernel) without
# writing outputs back to the .ipynb files.
set -euo pipefail
cd "$(dirname "$0")"

TMPDIR_NB=$(mktemp -d)
trap 'rm -rf "$TMPDIR_NB"' EXIT

for nb in 01_structured_amazon_products.ipynb 02_unstructured_amazon_reviews.ipynb; do
    echo ">>> Executing $nb"
    python -m nbconvert --to notebook --execute \
        --output-dir "$TMPDIR_NB" --output "${nb%.ipynb}.executed.ipynb" \
        "$nb"
done

echo ">>> Launching dashboard"
exec python -m streamlit run app.py
