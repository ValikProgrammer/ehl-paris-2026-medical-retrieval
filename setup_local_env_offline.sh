#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$ROOT_DIR/.uv-python/cpython-3.12.13-macos-aarch64-none/bin/python3"
VENV_DIR="$ROOT_DIR/.venv"
SITE_PACKAGES="$VENV_DIR/lib/python3.12/site-packages"
PTH_FILE="$SITE_PACKAGES/offline_uv_cache.pth"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"

mkdir -p "$SITE_PACKAGES"
cat > "$PTH_FILE" <<EOF_PTH
$ROOT_DIR/.uv-cache/archive-v0/tYxAnlR-PHOYOjVx
$ROOT_DIR/.uv-cache/archive-v0/7oIFkHFGaIMfOMAT
$ROOT_DIR/.uv-cache/archive-v0/HZuN95fQGTXQEtxB
$ROOT_DIR/.uv-cache/archive-v0/jHPvpCUirkyF2HMS
$ROOT_DIR/.uv-cache/archive-v0/yrcV7qWhcCaaHnrM
$ROOT_DIR/.uv-cache/archive-v0/LfRVKvonrDI2xP--
$ROOT_DIR/.uv-cache/archive-v0/Rbtz78W4Z3K9h3-a
$ROOT_DIR/.uv-cache/archive-v0/DxGCgPviVDhL-Tf8
$ROOT_DIR/.uv-cache/archive-v0/dzShcpwRYiRQXpTG
EOF_PTH

echo "offline environment ready at $VENV_DIR"
echo "activate with: source .venv/bin/activate"
