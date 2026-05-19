#!/usr/bin/env bash
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
# ─────────────────────────────────────────────────────────────
# Sifmography Infinite Transcriber — Launcher
# Double-click this file to start the app. Close to stop.
# ─────────────────────────────────────────────────────────────

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Clean up terminal title
echo -ne "\033]0;✨ Sifmography Infinite Transcriber\007"

# Check venv exists
VENV_PYTHON="$(dirname "$DIR")/Voice-Clone-Studio/venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
  echo "❌  Voice-Clone-Studio Virtual environment not found."
  echo "    Please make sure the environment is installed."
  echo ""
  read -p "Press Enter to close..."
  exit 1
fi

# Kill any leftover process on port 8090
EXISTING=$(lsof -ti:8090 2>/dev/null)
if [ -n "$EXISTING" ]; then
  echo "⚠  Cleaning up port 8090..."
  echo "$EXISTING" | xargs kill -9 2>/dev/null
  sleep 1
fi

echo ""
echo "✨  Sifmography Infinite Transcriber"
echo "──────────────────────────────────────────────"
echo "📍  Launching server at http://localhost:8090"
echo "🧠  Powered locally by Apple Silicon MLX & Whisper"
echo "──────────────────────────────────────────────"
echo ""

# Start server using the specified virtual environment python
"$VENV_PYTHON" -u "$DIR/transcribe_app.py" &
SERVER_PID=$!

# Wait for server to respond
echo -n "    Booting local UI engine"
for i in $(seq 1 15); do
  sleep 1
  echo -n "."
  STATUS=$(curl -s -I http://localhost:8090 2>/dev/null | head -n 1)
  if echo "$STATUS" | grep -q "200 OK"; then
    break
  fi
done
echo ""
echo ""

# Open local web app
open http://localhost:8090

echo "✅  Server is running and listening at http://localhost:8090"
echo "💡  Keep this window open while using the app."
echo "🛑  Close this window to shut down the server."
echo ""

# Setup trap to clean up server on exit
trap "echo ''; echo '🛑 Shutting down server...'; kill $SERVER_PID 2>/dev/null; exit 0" INT TERM EXIT

wait $SERVER_PID
