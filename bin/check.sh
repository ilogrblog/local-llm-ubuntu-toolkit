#!/usr/bin/env bash
# Runs the toolkit's own checks on this machine. Needs Python 3.10 or later.
# Reads files in this repository and talks only to stub servers it starts on
# 127.0.0.1. It installs nothing, writes nothing outside a temporary folder and
# never touches a running model server or a systemd unit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found. Install Python 3.10 or later, or set PYTHON=/path/to/python." >&2
  exit 2
fi
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "error: $("$PYTHON" -V 2>&1) is too old. This toolkit needs Python 3.10 or later." >&2
  exit 2
fi

fail=0
expect() { # expect <name> <wanted> <actual>
  if [ "$3" = "$2" ]; then
    echo "PASS  $1"
  else
    echo "FAIL  $1 (wanted '$2', got '$3')"
    fail=1
  fi
}
status() { # status <command...>: run it, print its exit status, hide its output
  local code=0
  "$@" >/dev/null 2>&1 || code=$?
  echo "$code"
}

echo "Python: $("$PYTHON" -V 2>&1)"

if "$PYTHON" -m unittest discover -s tests >/dev/null 2>&1; then
  echo "PASS  unit tests"
else
  echo "FAIL  unit tests (run: $PYTHON -m unittest discover -s tests -v)"
  fail=1
fi

# The two commands the guide prints, run through the real command line.
plan() { "$PYTHON" vram_plan.py --spec models.example.json --device 24GiB --reserve 2GiB "$@"; }
summary() { # summary <planner args...>: "fits max_sessions max_context"
  plan "$@" 2>/dev/null | "$PYTHON" -c '
import json, sys
r = json.load(sys.stdin)
limits = r["limits_at_this_reserve"]
print(r["fits"], limits["max_concurrency_at_requested_context"], limits["max_context_at_requested_concurrency"])' || true
}

expect "planner: 4 sessions at 8,192 tokens fit (up to 17 sessions or 35,671 tokens)" \
  "True 17 35671" "$(summary --context 8192 --concurrency 4)"
expect "planner: 8 sessions at 32,768 tokens do not fit (up to 4 sessions or 17,835 tokens)" \
  "False 4 17835" "$(summary --context 32768 --concurrency 8)"
expect "planner exits 1 when the configuration does not fit" \
  "1" "$(status plan --context 32768 --concurrency 8)"
expect "planner exits 2 on an unreadable spec" \
  "2" "$(status "$PYTHON" vram_plan.py --spec /nonexistent.json --device 1GiB --reserve 0 --context 1 --concurrency 1)"
expect "probe refuses a non-local host without --allow-remote" \
  "2" "$(status "$PYTHON" probe.py --model x --prompt-file /nonexistent.txt --base-url http://model.example.test)"
expect "probe --help runs" "0" "$(status "$PYTHON" probe.py --help)"

if [ "$fail" = 0 ]; then echo "All checks passed."; else echo "Some checks failed."; fi
exit "$fail"
