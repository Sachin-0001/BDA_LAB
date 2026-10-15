#!/usr/bin/env bash
# Download the real Lending Club dataset from Kaggle into data/.
# Needs a Kaggle API token: https://www.kaggle.com/settings -> "Create New Token" -> save as ~/.kaggle/kaggle.json
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f "$HOME/.kaggle/kaggle.json" ] || { echo "Missing ~/.kaggle/kaggle.json (see header of this script)"; exit 1; }
.venv/bin/pip install -q kaggle 2>/dev/null || ~/.local/bin/uv pip install --python .venv/bin/python -q kaggle
mkdir -p data/lending_club && cd data/lending_club
../../.venv/bin/kaggle datasets download -d wordsforthewise/lending-club --unzip
ls -lh
echo "Use the accepted_2007_to_2018Q4.csv(.gz) file:  make load CSV=data/lending_club/<file>"
