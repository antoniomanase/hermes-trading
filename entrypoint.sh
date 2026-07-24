#!/usr/bin/env bash
# Seed the persistent Railway volume (mounted empty at /app/state) from the
# image's baked defaults on first boot. On every subsequent boot the evolved
# state already exists, so nothing is overwritten.
set -e

mkdir -p /app/state /app/state/history

if [ -d /app/state_default ]; then
  for f in goal.yaml strategy.yaml; do
    if [ ! -f "/app/state/$f" ] && [ -f "/app/state_default/$f" ]; then
      cp "/app/state_default/$f" "/app/state/$f"
      echo "[entrypoint] seeded /app/state/$f from image default"
    fi
  done
  if [ -d /app/state_default/history ]; then
    cp -rn /app/state_default/history/. /app/state/history/ 2>/dev/null || true
  fi
fi

# Ensure append-only logs exist so the worker can open them immediately.
for f in trades.jsonl hypotheses.jsonl; do
  [ -f "/app/state/$f" ] || : > "/app/state/$f"
done

exec "$@"
