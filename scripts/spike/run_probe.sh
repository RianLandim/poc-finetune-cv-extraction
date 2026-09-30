#!/usr/bin/env bash
# usage: run_probe.sh <name> <vram_probe args...>  -- one config per process, log to file.
name=$1; shift
log=outputs/spike/p-$name.log
uv run python scripts/spike/vram_probe.py "$@" > "$log" 2>&1; ec=$?
r=$(grep '^RESULT ' "$log" | sed 's/^RESULT //')
if [ -n "$r" ]; then echo "$name: $r"; else echo "$name: FAIL exit=$ec :: $(grep -E 'Error|error' "$log" | tail -2)"; fi
