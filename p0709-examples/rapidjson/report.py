#!/usr/bin/env python3
"""Summarizes the results of run.sh as Markdown (see README.md).

Usage: report.py <work dir> "<configs>" <variants...>; the first variant is
the baseline.
"""
import bisect
import os
import re
import subprocess
import sys

work, configs, variants = sys.argv[1], sys.argv[2].split(), sys.argv[3:]
base = variants[0]
TITLES = {
    "default": "default code layout (-O2)",
    "aligned": "loops aligned to 64 bytes (-O2 -falign-loops=64)",
}


def load(config):
    results, checksums = {}, set()
    for v in variants:
        res = results.setdefault(v, {})
        for line in open(f"{work}/results/{config}/{v}.txt"):
            parts = line.strip().split(",")
            if parts[0] in ("result", "instructions"):
                key = (parts[0] if parts[0] == "instructions" else parts[1],
                       parts[2] if parts[0] == "result" else
                       f"{parts[1]} {parts[2]}")
                res.setdefault(key, []).append(float(parts[3]))
            elif parts[0] == "checksum":
                checksums.add(parts[1])
    return results, checksums


def table(results, title, section, unit, higher_is_better, fmt="{:.1f}"):
    keys = [k for k in results[base] if k[0] == section]
    if not keys:
        return
    runs = len(results[base][keys[0]])
    print(f"\n#### {title}\n\n{unit}"
          + (f", best of {runs} runs" if runs > 1 else "")
          + f"; in parentheses: change relative to `{base}`.\n")
    print("| | " + " | ".join(f"`{v}`" for v in variants) + " |")
    print("|---|" + "---:|" * len(variants))
    pick = max if higher_is_better else min
    for k in keys:
        b = pick(results[base][k])
        cells = []
        for v in variants:
            x = pick(results[v][k])
            cells.append(fmt.format(x) if v == base else
                         f"{fmt.format(x)} ({(x / b - 1) * 100:+.1f}%)")
        print(f"| {k[1]} | " + " | ".join(cells) + " |")


# The parser's functions, plus the callers into which parts of it are inlined
# (how much differs between the variants).
PARSER_SYMBOLS = ("GenericReader", "::ParseStream<", "parseSAX(")
DWARFDUMP = os.environ.get("DWARFDUMP", "llvm-dwarfdump")


def code_size(config):
    """Bytes of the parser's code, unwind info (its FDEs in .eh_frame) and
    exception tables (its LSDAs in .gcc_except_table), per variant."""
    sizes = []
    for v in variants:
        binary = f"{work}/bin/{config}/bench-{v}"
        funcs = []  # (address, size, name) of all functions
        for line in subprocess.run(["nm", "-S", "-C", binary],
                                   capture_output=True, text=True,
                                   check=True).stdout.splitlines():
            f = line.split(None, 3)
            if len(f) == 4 and f[2] in "tTwW":
                funcs.append((int(f[0], 16), int(f[1], 16), f[3]))
        funcs.sort()
        starts = [a for a, _, _ in funcs]

        def is_parser(pc):
            i = bisect.bisect_right(starts, pc) - 1
            return i >= 0 and any(n in funcs[i][2] for n in PARSER_SYMBOLS)

        text = sum(s for _, s, n in funcs
                   if any(p in n for p in PARSER_SYMBOLS))

        # FDEs: "<offset> <length> <cie ptr> FDE cie=... pc=<begin>...<end>",
        # followed by "LSDA Address: <address>" if the function has one.
        fdes, lsdas, current = 0, [], None
        for line in subprocess.run([DWARFDUMP, "--eh-frame", binary],
                                   capture_output=True, text=True,
                                   check=True).stdout.splitlines():
            m = re.match(r"[0-9a-f]+ ([0-9a-f]+) [0-9a-f]+ FDE .*pc=([0-9a-f]+)",
                         line)
            if m:
                current = is_parser(int(m.group(2), 16))
                fdes += (int(m.group(1), 16) + 4) * current
            m = re.match(r"\s+LSDA Address: ([0-9a-f]+)", line)
            if m:
                lsdas.append((int(m.group(1), 16), current))
        # An LSDA extends to the next one (or the end of the section).
        for line in subprocess.run(["readelf", "-SW", binary],
                                   capture_output=True, text=True,
                                   check=True).stdout.splitlines():
            f = line.split()
            if ".gcc_except_table" in f:
                i = f.index(".gcc_except_table")
                table_end = int(f[i + 2], 16) + int(f[i + 4], 16)
        lsdas.sort()
        ends = [a for a, _ in lsdas[1:]] + [table_end] if lsdas else []
        tables = sum(e - a for (a, parser), e in zip(lsdas, ends) if parser)
        sizes.append((text, fdes, tables))
    return sizes


def size_table(sizes):
    print("\n#### Size of the parser\n\nBytes of all `GenericReader` functions, "
          "`GenericDocument::ParseStream` and `parseSAX` (into which parts "
          "of the parser are inlined); in parentheses: change relative to "
          f"`{base}`.\n")
    print("| | " + " | ".join(f"`{v}`" for v in variants) + " |")
    print("|---|" + "---:|" * len(variants))
    rows = [("code (.text)", 0), ("unwind info (.eh_frame)", 1),
            ("exception tables (.gcc_except_table)", 2), ("total", None)]
    for title, i in rows:
        vals = [sum(s) if i is None else s[i] for s in sizes]
        cells = [str(x) if v == base or not vals[0] else
                 f"{x} ({(x / vals[0] - 1) * 100:+.1f}%)"
                 for v, x in zip(variants, vals)]
        bold = "**" if i is None else ""
        print(f"| {bold}{title}{bold} | " + " | ".join(cells) + " |")


print("## RapidJSON with P0709 static exceptions\n")
print(open(f"{work}/compiler.txt").read().strip())
all_checksums = set()
for config in configs:
    results, checksums = load(config)
    all_checksums |= checksums
    print(f"\n### {TITLES.get(config, config)}")
    table(results, "a) Happy path: DOM parsing", "dom",
          "MB/s, higher is better", True)
    table(results, "a) Happy path: SAX parsing", "sax",
          "MB/s, higher is better", True)
    table(results, "a) Happy path: retired instructions per parse",
          "instructions", "Instructions (perf stat), lower is better", False,
          "{:.0f}")
    table(results, "b) Errors: small documents, a fraction of them corrupted",
          "errors", "ns per document, lower is better", False)
    size_table(code_size(config))

if len(all_checksums) == 1:
    print(f"\nAll variants computed the same results "
          f"(checksum {all_checksums.pop()}).")
else:
    print(f"\n**WARNING: the results differ between variants:** "
          f"{sorted(all_checksums)}")
