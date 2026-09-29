# RapidJSON with P0709 static exceptions

This experiment measures the cost of the error handling mechanisms of the
P0709 prototype (`-fstatic-exceptions`) in a real, well-known C++ program:
the parser of [RapidJSON](https://github.com/Tencent/rapidjson), with the
standard inputs of
[nativejson-benchmark](https://github.com/miloyip/nativejson-benchmark).

It answers two questions:

a) **Happy path:** how much does error handling cost when no errors occur?
b) **Errors:** how much does it cost when errors do occur?

## Reproducing

```sh
CPU=3 ./run.sh          # needs git, curl, python3; perf is optional
```

`run.sh` clones RapidJSON and downloads the inputs at fixed versions into
`work/`, builds all variants with the prototype compiler
(`../../build/bin/clang++`, or `$CXX`), runs them `$ROUNDS` times (default
5), interleaved, and writes `work/report.md`. `CPU` pins the benchmarks to a
core. If `perf stat` works, the retired instructions of each parse are
counted as well.

## Variants

RapidJSON's recursive-descent parser reports an error by recording the code
and offset in the reader and returning; each caller checks for an error
after every call that can fail (`RAPIDJSON_PARSE_ERROR_EARLY_RETURN`).
`patch-rapidjson.py` makes a small, mechanical change to the headers: it
marks the 11 functions of `reader.h` that can fail with `RAPIDJSON_THROWS`,
and it wraps the public `GenericReader::Parse` in a `try`/`catch`
(`RAPIDJSON_P0709_CATCH`), so that the rest of the API (e.g.
`Document::Parse`) is unchanged. It also marks what the parser calls (the
`Document` handler, the stacks, the pool allocator, UTF-8 encoding)
`noexcept`, which it is in fact: RapidJSON never throws. The experiment
compares worlds that use a single error handling mechanism; without this,
every `throws` function would have to catch and translate dynamic
exceptions (e.g. `std::bad_alloc`) from these calls.
`config.h` then selects the error handling:

| variant | error reporting |
|---|---|
| `codes` | the original: error codes, checked after each call (`RAPIDJSON_THROWS` is `noexcept`) |
| `dynamic` | traditional C++ exceptions (`throw`, table-based unwinding) |
| `static-pointer` | `throws`, `-fstatic-exceptions-abi=pointer` (hidden error pointer) |
| `static-register` | `throws`, `-fstatic-exceptions-abi=register` (the default) |
| `static-carry` | `throws`, `-fstatic-exceptions-abi=carry` (flag in the x86 carry flag) |

In all variants the error code and offset are recorded before the error is
reported, so all of them compute the same results (the report checks this
with a checksum).

## Benchmarks

`bench.cpp`:

a) **Happy path:** parses `canada.json` (2.2 MB, mostly numbers),
   `citm_catalog.json` (1.7 MB, nested objects) and `twitter.json` (0.6 MB,
   mostly strings) into a DOM (`Document::Parse`) and with a SAX handler that
   only counts events. It reports MB/s, and (with perf) retired instructions
   per parse.
b) **Errors:** parses a corpus of 284 small documents (the tweets of
   `twitter.json` and the events of `citm_catalog.json`, about 1.8 KB on
   average) into a DOM each; a given fraction of the documents has a control
   character at a random position, which makes the parse fail there. It
   reports ns per document.

## Measuring pitfall: code layout

The timings of the happy path are dominated by a few hot loops (e.g. the
digit loop of `ParseNumber` for `canada.json`), and on the machine used
here their speed depends strongly on where they happen to be placed in
memory: unrelated changes to the benchmark driver changed the DOM throughput
of `canada.json` of one and the same variant by up to 40%, in either
direction, with the same instructions executed. Therefore the report shows

- the default build (`-O2`), whose timings contain such layout effects,
- a build with all loops aligned to 64 bytes (`-O2 -falign-loops=64`),
  which removes most of them, and
- the number of retired instructions per parse, which does not depend on
  the layout and directly shows the cost of the additional checks.

Conclusions about the happy path should be drawn from the latter two.

## Results

See `results.md` for the results of a run on the authors' machine.
