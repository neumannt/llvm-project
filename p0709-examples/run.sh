#!/bin/sh
# Builds and runs the P0709 static exceptions examples with the prototype
# compiler at several optimization levels, then runs the benchmark.
set -e
CXX=${CXX:-"$(dirname "$0")/../build/bin/clang++"}
cd "$(dirname "$0")"
out=$(mktemp -d)
for opt in -O0 -O2; do
  for f in paper cleanups interop advanced cond sugar misc; do
    printf '%-10s %s: ' "$f" "$opt"
    "$CXX" -std=c++17 -fstatic-exceptions -Wno-unused-result $opt $f.cpp -o "$out/$f"
    "$out/$f" | tail -1
  done
  printf '%-10s %s: ' noexc "$opt"
  "$CXX" -std=c++17 -fstatic-exceptions -fno-exceptions $opt noexc.cpp -o "$out/noexc"
  "$out/noexc" | tail -1
  printf '%-10s %s: ' hook "$opt"
  "$CXX" -std=c++17 -fstatic-exceptions -fstatic-exceptions-propagation-hook $opt hook.cpp -o "$out/hook"
  "$out/hook" | tail -1
done
"$CXX" -std=c++17 -fstatic-exceptions -O2 bench.cpp -o "$out/bench"
"$out/bench"
rm -rf "$out"
