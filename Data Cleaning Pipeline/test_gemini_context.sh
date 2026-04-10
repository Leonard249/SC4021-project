#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROMPT="Summarize the instruction you received in GEMINI.md in one paragraph."
RUNNER="$SCRIPT_DIR/openai_headless.py"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH." >&2
  exit 1
fi

if [ ! -f "$RUNNER" ]; then
  echo "Error: openai_headless.py was not found in $SCRIPT_DIR." >&2
  exit 1
fi

if [ ! -f "$SCRIPT_DIR/GEMINI.md" ]; then
  echo "Error: GEMINI.md was not found in $SCRIPT_DIR." >&2
  exit 1
fi

echo "Running stateless OpenAI-compatible prompt from: $SCRIPT_DIR"
echo "Prompt: $PROMPT"
echo

conda run -n sc4021 python "$RUNNER" -p "$PROMPT"
