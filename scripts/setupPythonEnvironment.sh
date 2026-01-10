#!/usr/bin/env bash
set -euo pipefail

# Navigate to repository root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ensure_path_line='export PATH="$HOME/.local/bin:$PATH"'

# Install uv if not already available
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Add to PATH for this script (common install path on Linux/WSL)
  export PATH="$HOME/.local/bin:$PATH"

  # Persist PATH change for future shells (only if needed)
  if [ ! -f "$HOME/.bashrc" ]; then
    touch "$HOME/.bashrc"
  fi
  if ! grep -Fxq "$ensure_path_line" "$HOME/.bashrc"; then
    echo "" >> "$HOME/.bashrc"
    echo "$ensure_path_line" >> "$HOME/.bashrc"
    echo "Added uv to PATH in ~/.bashrc"
  fi

  # Sanity check
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv installation completed, but 'uv' is still not on PATH."
    echo "Try restarting your shell, or run: source ~/.bashrc"
    exit 1
  fi
else
  echo "uv already installed"
fi

# Ensure uv is in PATH (in case installed but not in current PATH)
export PATH="$HOME/.local/bin:$PATH"

# Install Python (version from .python-version)
uv python install

# Create virtual environment and sync dependencies
uv sync --locked

# Verify Python installation
uv run python -V

echo ""
echo "Setup complete! If this was your first run, restart your shell or run:"
echo "  source ~/.bashrc"
