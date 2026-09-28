#!/bin/sh
# Builds abi-bench.cpp with each -fstatic-exceptions-abi, and with dynamic
# exceptions as a baseline, and runs it.
set -e
CXX=${CXX:-"$(dirname "$0")/../build/bin/clang++"}
cd "$(dirname "$0")"
out=$(mktemp -d)
"$CXX" -std=c++17 -O2 -DDYNAMIC_EXCEPTIONS abi-bench.cpp -o "$out/abi-bench-dynamic"
for abi in pointer register carry; do
  "$CXX" -std=c++17 -fstatic-exceptions -fstatic-exceptions-abi=$abi -O2 \
    abi-bench.cpp -o "$out/abi-bench-$abi"
done
for v in dynamic pointer register carry; do
  "$out/abi-bench-$v" $v
  echo
done
rm -rf "$out"
