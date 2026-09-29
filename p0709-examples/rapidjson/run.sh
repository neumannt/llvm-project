#!/bin/sh
# Reproduces the RapidJSON experiment for the P0709 static exceptions
# prototype (see README.md). Environment variables:
#   CXX     the prototype clang++ (default: ../../build/bin/clang++)
#   WORK    work directory for sources, data and binaries (default: ./work)
#   ROUNDS  number of interleaved runs of all variants (default: 5)
#   CPU     if set, pin the benchmarks to this CPU with taskset
#   PERF    set to 0 to skip counting instructions with perf stat
#   DWARFDUMP  llvm-dwarfdump, to measure unwind info (default: next to CXX)
set -e
here=$(cd "$(dirname "$0")" && pwd)
CXX=${CXX:-$here/../../build/bin/clang++}
WORK=${WORK:-$here/work}
ROUNDS=${ROUNDS:-5}
PERF=${PERF:-1}
RAPIDJSON_COMMIT=24b5e7a8b27f42fa16b96fc70aade9106cf7102f
DATA_COMMIT=478d5727c2a4048e835a29c65adecc7d795360d5
DATA_URL=https://raw.githubusercontent.com/miloyip/nativejson-benchmark/$DATA_COMMIT/data

# Two configurations: the default code layout, and with all loops aligned to
# 64 bytes, which removes most of the (large) effects of where hot loops
# happen to be placed.
CONFIGS="default aligned"
config_flags() {
  case $1 in
    default) echo -O2 ;;
    aligned) echo -O2 -falign-loops=64 ;;
  esac
}
VARIANTS="codes dynamic static-pointer static-register static-carry"
variant_flags() {
  case $1 in
    codes) echo -DRJ_CODES ;;
    dynamic) echo -DRJ_DYNAMIC ;;
    static-*) echo -DRJ_STATIC -fstatic-exceptions \
                   -fstatic-exceptions-abi=${1#static-} ;;
  esac
}

mkdir -p "$WORK/data"
cd "$WORK"

# Sources and inputs, at fixed versions.
if [ ! -d rapidjson ]; then
  git clone -q https://github.com/Tencent/rapidjson.git
fi
git -C rapidjson checkout -q "$RAPIDJSON_COMMIT"
git -C rapidjson checkout -q -- include
python3 "$here/patch-rapidjson.py" rapidjson/include/rapidjson
for f in canada citm_catalog twitter; do
  [ -f data/$f.json ] || curl -sSfL -o data/$f.json "$DATA_URL/$f.json"
done

pin=
[ -n "$CPU" ] && pin="taskset -c $CPU"
if [ "$PERF" != 0 ] && ! perf stat -x, -e instructions true >/dev/null 2>&1; then
  echo "perf stat does not work here; not counting instructions" >&2
  PERF=0
fi

echo "compiler: $("$CXX" --version | head -1)" > compiler.txt
for c in $CONFIGS; do
  mkdir -p bin/$c results/$c
  rm -f results/$c/*.txt
  for v in $VARIANTS; do
    "$CXX" -std=c++17 $(config_flags $c) -DNDEBUG $(variant_flags $v) \
      -include "$here/config.h" -I rapidjson/include "$here/bench.cpp" \
      -o bin/$c/bench-$v
  done

  # Timings, with the variants interleaved.
  r=1
  while [ $r -le "$ROUNDS" ]; do
    for v in $VARIANTS; do
      echo "$c: round $r/$ROUNDS: $v" >&2
      $pin bin/$c/bench-$v data >> results/$c/$v.txt
    done
    r=$((r + 1))
  done

  # Retired instructions per parse (independent of code layout): the
  # difference between 21 and 1 parses, divided by 20.
  if [ "$PERF" != 0 ]; then
    for v in $VARIANTS; do
      for mode in dom sax; do
        for f in canada.json citm_catalog.json twitter.json; do
          count() {
            $pin perf stat -x, -e instructions:u \
              bin/$c/bench-$v data repeat $f $mode $1 2>&1 >/dev/null |
              grep instructions | cut -d, -f1
          }
          echo "instructions,$mode,$f,$(( ($(count 21) - $(count 1)) / 20 ))" \
            >> results/$c/$v.txt
        done
      done
    done
  fi
done

DWARFDUMP=${DWARFDUMP:-$(dirname "$CXX")/llvm-dwarfdump} \
  python3 "$here/report.py" "$WORK" "$CONFIGS" $VARIANTS | tee report.md
