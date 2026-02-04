#!/bin/bash
# Export marimo examples to HTML for documentation

set -e

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$DOCS_DIR")"

echo "Exporting marimo examples to docs/examples/"

# Export each example
for notebook in "$ROOT_DIR"/examples/*.py; do
    if [[ -f "$notebook" && "$(basename "$notebook")" != "__"* ]]; then
        name=$(basename "$notebook" .py)
        echo "  Exporting $name..."
        marimo export html "$notebook" -o "$DOCS_DIR/examples/${name}.html" --include-code
    fi
done

echo "Done! Run 'mkdocs serve' to preview."
