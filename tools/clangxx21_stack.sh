#!/usr/bin/env bash
ulimit -s unlimited 2>/dev/null || true
if [[ -x /usr/lib/llvm-21/bin/clang++ ]]; then
    exec /usr/lib/llvm-21/bin/clang++ "$@"
fi
if command -v clang++-21 >/dev/null 2>&1; then
    exec clang++-21 "$@"
fi
exec clang++ "$@"
