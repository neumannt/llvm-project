# Results

Measured on an AMD Ryzen 9 9950X3D (Zen 5), Linux 7.1, pinned to one core
(`CPU=3 ./run.sh`, 5 interleaved rounds). The full generated report follows
the summary.

## Summary

**a) Happy path.** Retired instructions are the layout-independent measure
of the overhead:

- `static-register` executes 0.6–1.1% more instructions than the original
  error codes for DOM parsing, and 0.7–2.6% more for SAX parsing.
  `static-carry` executes 0.3–0.9% more for DOM and 0.2–2.4% more for SAX.
  `static-pointer` executes 0.3–2.9% more.
- Traditional exceptions (`dynamic`) mostly execute 1–2% fewer
  instructions than the error codes (0.7% more for SAX parsing of
  `canada.json`), as they need no checks at all. This is the zero-overhead
  reference: `static-register` executes 1.6–3.1% more instructions than
  it, `static-carry` 1.3–2.9% more.
- With the code layout effects removed (loops aligned), the throughput of
  `static-register` and `static-carry` is within −3.4% and +7.8% of the
  error codes, which is about the noise level (the variants differ by
  similar amounts among themselves, and `dynamic` by −1.8% to +2.7%).
  `static-pointer` is 23–25% slower for DOM parsing of `canada.json`
  with both code layouts, although it executes only 2.1–2.5% more
  instructions; the cause (e.g. passing the error through memory in the
  number parsing loop) has not been investigated.
- With the default layout, the timings vary by up to 26% depending on
  where the hot loops happen to be placed (e.g. `codes` is 25% faster than
  `static-register` for DOM parsing of `canada.json`, but 3% slower with
  aligned loops). These differences are not caused by the error handling
  (see the README).
- Size of the parser: with the register ABIs, the code grows by 0.1–1.5%,
  and the code plus unwind info by −0.1% to +1.2%; the static variants
  need no exception tables. With traditional exceptions, the code grows
  by 5.6–6.3%, and with unwind info and exception tables by 8.5–9.3%.
  `static-pointer` is in between (2.0–4.4%). (The static variants inline
  less of the parser into its callers, so the callers are counted as well.)

In all variants, the functions that the parser calls (e.g. the `Document`
handler) are `noexcept`, as they would be in a world that uses only one
error handling mechanism. Without this, each `throws` function had to
catch and translate dynamic exceptions from these calls: in an earlier
run, this made the code of the static variants 5–10% larger and gave
them larger exception tables than `dynamic`.

**b) Errors.** With `throws`, the cost of a failing document stays that of
the error codes, whatever the error rate (`static-register`: within ±0.6%
of the error codes). Traditional exceptions cost (with aligned loops) 6%
more at 1% failed documents, 16% more at 10%, 51% more at 50%, and 2.2
times as much at 100%. (With the default layout, `codes` and
`static-register`, and `static-pointer` from 1% on, are 14–15% slower
than with aligned loops, which distorts the comparison.)

In short: in this parser, static exceptions with the register ABI have
about the cost of hand-written error codes on the happy path, i.e. a few
percent at most over traditional exceptions, with about the same code
size as error codes and none of the size overhead of traditional
exceptions, and they keep that cost when errors occur, where traditional
exceptions become expensive.

## Generated report

compiler: clang version 24.0.0git (https://github.com/llvm/llvm-project.git a3c759e596f5102c591caa4040895b7fad41a128)

### default code layout (-O2)

#### a) Happy path: DOM parsing

MB/s, higher is better, best of 5 runs; in parentheses: change relative to `codes`.

| | `codes` | `dynamic` | `static-pointer` | `static-register` | `static-carry` |
|---|---:|---:|---:|---:|---:|
| canada.json | 1247.9 | 1268.5 (+1.7%) | 956.7 (-23.3%) | 931.4 (-25.4%) | 928.5 (-25.6%) |
| citm_catalog.json | 2156.8 | 2227.8 (+3.3%) | 2156.4 (-0.0%) | 2144.6 (-0.6%) | 2214.2 (+2.7%) |
| twitter.json | 1216.2 | 1429.0 (+17.5%) | 1212.2 (-0.3%) | 1322.1 (+8.7%) | 1393.7 (+14.6%) |

#### a) Happy path: SAX parsing

MB/s, higher is better, best of 5 runs; in parentheses: change relative to `codes`.

| | `codes` | `dynamic` | `static-pointer` | `static-register` | `static-carry` |
|---|---:|---:|---:|---:|---:|
| canada.json | 1623.5 | 1351.7 (-16.7%) | 1646.2 (+1.4%) | 1610.1 (-0.8%) | 1606.6 (-1.0%) |
| citm_catalog.json | 2173.1 | 2201.9 (+1.3%) | 2340.6 (+7.7%) | 2337.3 (+7.6%) | 2339.3 (+7.6%) |
| twitter.json | 1266.3 | 1388.7 (+9.7%) | 1479.9 (+16.9%) | 1478.0 (+16.7%) | 1300.3 (+2.7%) |

#### a) Happy path: retired instructions per parse

Instructions (perf stat), lower is better; in parentheses: change relative to `codes`.

| | `codes` | `dynamic` | `static-pointer` | `static-register` | `static-carry` |
|---|---:|---:|---:|---:|---:|
| dom canada.json | 65152913 | 63926543 (-1.9%) | 66768800 (+2.5%) | 65876750 (+1.1%) | 65764616 (+0.9%) |
| dom citm_catalog.json | 26922742 | 26439355 (-1.8%) | 27677401 (+2.8%) | 27096932 (+0.6%) | 26998949 (+0.3%) |
| dom twitter.json | 14563671 | 14392619 (-1.2%) | 14918174 (+2.4%) | 14655877 (+0.6%) | 14605559 (+0.3%) |
| sax canada.json | 48609301 | 48957599 (+0.7%) | 49292566 (+1.4%) | 49737539 (+2.3%) | 49792583 (+2.4%) |
| sax citm_catalog.json | 20173069 | 19992425 (-0.9%) | 20405411 (+1.2%) | 20385416 (+1.1%) | 20299343 (+0.6%) |
| sax twitter.json | 11602486 | 11482378 (-1.0%) | 11645871 (+0.4%) | 11680344 (+0.7%) | 11630595 (+0.2%) |

#### b) Errors: small documents, a fraction of them corrupted

ns per document, lower is better, best of 5 runs; in parentheses: change relative to `codes`.

| | `codes` | `dynamic` | `static-pointer` | `static-register` | `static-carry` |
|---|---:|---:|---:|---:|---:|
| 0% | 1970.9 | 1701.6 (-13.7%) | 1782.2 (-9.6%) | 1975.0 (+0.2%) | 1723.8 (-12.5%) |
| 1% | 1948.3 | 1739.4 (-10.7%) | 1952.3 (+0.2%) | 1946.1 (-0.1%) | 1704.5 (-12.5%) |
| 10% | 1821.3 | 1724.9 (-5.3%) | 1835.8 (+0.8%) | 1829.2 (+0.4%) | 1598.1 (-12.3%) |
| 50% | 1443.3 | 1795.9 (+24.4%) | 1453.5 (+0.7%) | 1449.6 (+0.4%) | 1264.3 (-12.4%) |
| 100% | 997.7 | 1834.8 (+83.9%) | 1005.8 (+0.8%) | 1001.9 (+0.4%) | 875.6 (-12.2%) |

#### Size of the parser

Bytes of all `GenericReader` functions, `GenericDocument::ParseStream` and `parseSAX` (into which parts of the parser are inlined); in parentheses: change relative to `codes`.

| | `codes` | `dynamic` | `static-pointer` | `static-register` | `static-carry` |
|---|---:|---:|---:|---:|---:|
| code (.text) | 15676 | 16663 (+6.3%) | 16368 (+4.4%) | 15833 (+1.0%) | 15918 (+1.5%) |
| unwind info (.eh_frame) | 1292 | 1500 (+16.1%) | 1340 (+3.7%) | 1268 (-1.9%) | 1248 (-3.4%) |
| exception tables (.gcc_except_table) | 0 | 384 | 0 | 0 | 0 |
| **total** | 16968 | 18547 (+9.3%) | 17708 (+4.4%) | 17101 (+0.8%) | 17166 (+1.2%) |

### loops aligned to 64 bytes (-O2 -falign-loops=64)

#### a) Happy path: DOM parsing

MB/s, higher is better, best of 5 runs; in parentheses: change relative to `codes`.

| | `codes` | `dynamic` | `static-pointer` | `static-register` | `static-carry` |
|---|---:|---:|---:|---:|---:|
| canada.json | 1229.0 | 1228.4 (-0.0%) | 922.1 (-25.0%) | 1190.9 (-3.1%) | 1187.6 (-3.4%) |
| citm_catalog.json | 2191.2 | 2250.0 (+2.7%) | 2158.4 (-1.5%) | 2171.6 (-0.9%) | 2208.4 (+0.8%) |
| twitter.json | 1402.9 | 1411.0 (+0.6%) | 1398.8 (-0.3%) | 1400.7 (-0.2%) | 1397.2 (-0.4%) |

#### a) Happy path: SAX parsing

MB/s, higher is better, best of 5 runs; in parentheses: change relative to `codes`.

| | `codes` | `dynamic` | `static-pointer` | `static-register` | `static-carry` |
|---|---:|---:|---:|---:|---:|
| canada.json | 1635.8 | 1606.6 (-1.8%) | 1665.1 (+1.8%) | 1606.1 (-1.8%) | 1603.3 (-2.0%) |
| citm_catalog.json | 2212.7 | 2216.0 (+0.1%) | 2356.3 (+6.5%) | 2206.3 (-0.3%) | 2341.5 (+5.8%) |
| twitter.json | 1365.2 | 1376.5 (+0.8%) | 1425.4 (+4.4%) | 1377.5 (+0.9%) | 1472.0 (+7.8%) |

#### a) Happy path: retired instructions per parse

Instructions (perf stat), lower is better; in parentheses: change relative to `codes`.

| | `codes` | `dynamic` | `static-pointer` | `static-register` | `static-carry` |
|---|---:|---:|---:|---:|---:|
| dom canada.json | 65987368 | 65040233 (-1.4%) | 67380997 (+2.1%) | 66711195 (+1.1%) | 66599089 (+0.9%) |
| dom citm_catalog.json | 27112724 | 26633820 (-1.8%) | 27887880 (+2.9%) | 27286914 (+0.6%) | 27188931 (+0.3%) |
| dom twitter.json | 14626083 | 14454943 (-1.2%) | 15005484 (+2.6%) | 14718289 (+0.6%) | 14667971 (+0.3%) |
| sax canada.json | 49555378 | 49903680 (+0.7%) | 50294221 (+1.5%) | 50850829 (+2.6%) | 50738702 (+2.4%) |
| sax citm_catalog.json | 20552806 | 20212949 (-1.7%) | 20761272 (+1.0%) | 20802928 (+1.2%) | 20730816 (+0.9%) |
| sax twitter.json | 11758968 | 11547972 (-1.8%) | 11799096 (+0.3%) | 11850736 (+0.8%) | 11813765 (+0.5%) |

#### b) Errors: small documents, a fraction of them corrupted

ns per document, lower is better, best of 5 runs; in parentheses: change relative to `codes`.

| | `codes` | `dynamic` | `static-pointer` | `static-register` | `static-carry` |
|---|---:|---:|---:|---:|---:|
| 0% | 1726.1 | 1692.2 (-2.0%) | 1729.3 (+0.2%) | 1720.5 (-0.3%) | 1721.0 (-0.3%) |
| 1% | 1707.3 | 1812.3 (+6.2%) | 1703.3 (-0.2%) | 1707.7 (+0.0%) | 1713.9 (+0.4%) |
| 10% | 1600.8 | 1848.9 (+15.5%) | 1597.3 (-0.2%) | 1598.3 (-0.2%) | 1614.3 (+0.8%) |
| 50% | 1256.5 | 1895.7 (+50.9%) | 1263.6 (+0.6%) | 1264.2 (+0.6%) | 1275.3 (+1.5%) |
| 100% | 868.6 | 1934.7 (+122.7%) | 871.8 (+0.4%) | 869.3 (+0.1%) | 882.1 (+1.6%) |

#### Size of the parser

Bytes of all `GenericReader` functions, `GenericDocument::ParseStream` and `parseSAX` (into which parts of the parser are inlined); in parentheses: change relative to `codes`.

| | `codes` | `dynamic` | `static-pointer` | `static-register` | `static-carry` |
|---|---:|---:|---:|---:|---:|
| code (.text) | 17191 | 18162 (+5.6%) | 17540 (+2.0%) | 17204 (+0.1%) | 17289 (+0.6%) |
| unwind info (.eh_frame) | 1292 | 1500 (+16.1%) | 1340 (+3.7%) | 1268 (-1.9%) | 1248 (-3.4%) |
| exception tables (.gcc_except_table) | 0 | 388 | 0 | 0 | 0 |
| **total** | 18483 | 20050 (+8.5%) | 18880 (+2.1%) | 18472 (-0.1%) | 18537 (+0.3%) |

All variants computed the same results (checksum 3d590ce19c05452f).
