#!/bin/bash
# One-time pixi bootstrap on a Leonardo LOGIN node (has internet).
# Run: ~/.pi/agent/skills/leonardo-connect/scripts/leo.sh run "$(cat bootstrap_pixi.sh)"
set -euo pipefail

if ! command -v pixi >/dev/null 2>&1 && [ ! -x "$HOME/.pixi/bin/pixi" ]; then
  echo "Installing pixi..."
  curl -fsSL https://pixi.sh/install.sh | bash
fi
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version

PROJ="$HOME/zero-one"
if [ ! -f "$PROJ/pixi.toml" ]; then
  mkdir -p "$PROJ" && cd "$PROJ"
  pixi init . || true
  pixi add python pytorch       # conda-forge; add cuda build as needed
  # pixi add --pypi <pkg>        # PyPI packages
fi
echo "pixi ready at $PROJ — run: pixi run --manifest-path $PROJ/pixi.toml python3 -c 'print(1)'"
