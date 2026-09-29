#!/usr/bin/env python3
"""Prepares RapidJSON's headers for the P0709 experiment.

The recursive parser reports errors through the RAPIDJSON_PARSE_ERROR*
macros, which config.h redefines per variant. For static exceptions, the
functions that can fail must also be declared 'throws'. This script

  - marks those functions in reader.h with RAPIDJSON_THROWS (empty unless
    config.h defines it),
  - renames GenericReader::Parse to ParseImpl and adds a Parse wrapper that
    catches the exception (RAPIDJSON_P0709_CATCH), so that the public API,
    e.g. Document::Parse, is unchanged, and
  - marks the functions that the parser calls and that are not inlined, or
    that are inlined but call such functions (the Document handler, the
    stacks, the pool allocator, UTF-8 encoding), noexcept.

The latter is what they are in fact: RapidJSON never throws, and reports
allocation failures (if at all) by null pointers. The experiment compares
worlds that use only one error handling mechanism; without noexcept, each
'throws' function would have to catch and translate dynamic exceptions
(e.g. std::bad_alloc) from these calls. The noexcept is added in all
variants, so that they compile the same source.

Without config.h, the patched headers behave like the originals.
Usage: patch-rapidjson.py <path to rapidjson/include/rapidjson>
"""
import sys

include = sys.argv[1]


def patch(file, replacements):
    path = f"{include}/{file}"
    src = open(path).read()
    for old, new, *count in replacements:
        count = count[0] if count else 1
        found = src.count(old)
        if found != count:
            sys.exit(f"patch-rapidjson.py: {file}: expected {count} "
                     f"occurrence(s), found {found}:\n{old}")
        src = src.replace(old, new)
    open(path, "w").write(src)


# reader.h: 'throws' for the functions of the recursive parser that can fail.
reader = [("#define RAPIDJSON_NOTHING /* deliberately empty */\n",
           "#define RAPIDJSON_NOTHING /* deliberately empty */\n"
           "#ifndef RAPIDJSON_THROWS\n"
           "#define RAPIDJSON_THROWS /* P0709 experiment: see config.h */\n"
           "#endif\n")]
for sig in [
    "void SkipWhitespaceAndComments(InputStream& is)",
    "void ParseObject(InputStream& is, Handler& handler)",
    "void ParseArray(InputStream& is, Handler& handler)",
    "void ParseNull(InputStream& is, Handler& handler)",
    "void ParseTrue(InputStream& is, Handler& handler)",
    "void ParseFalse(InputStream& is, Handler& handler)",
    "unsigned ParseHex4(InputStream& is, size_t escapeOffset)",
    "void ParseString(InputStream& is, Handler& handler, bool isKey = false)",
    "void ParseStringToStream(InputStream& is, OutputStream& os)",
    "void ParseNumber(InputStream& is, Handler& handler)",
    "void ParseValue(InputStream& is, Handler& handler)",
]:
    reader.append((sig + " {", sig + " RAPIDJSON_THROWS {"))

# reader.h: Parse -> ParseImpl, plus a wrapper that catches.
reader.append(("""    template <unsigned parseFlags, typename InputStream, typename Handler>
    ParseResult Parse(InputStream& is, Handler& handler) {
        if (parseFlags & kParseIterativeFlag)""",
               """    template <unsigned parseFlags, typename InputStream, typename Handler>
    ParseResult Parse(InputStream& is, Handler& handler) {
#ifdef RAPIDJSON_P0709_CATCH
        try {
            return ParseImpl<parseFlags>(is, handler);
        } RAPIDJSON_P0709_CATCH {
            return parseResult_;
        }
#else
        return ParseImpl<parseFlags>(is, handler);
#endif
    }

    template <unsigned parseFlags, typename InputStream, typename Handler>
    ParseResult ParseImpl(InputStream& is, Handler& handler) RAPIDJSON_THROWS {
        if (parseFlags & kParseIterativeFlag)"""))
patch("reader.h", reader)

# noexcept for what the parser calls.
patch("internal/stack.h", [
    ("    void Expand(size_t count) {",
     "    void Expand(size_t count) RAPIDJSON_NOEXCEPT {"),
    ("    void Resize(size_t newCapacity) {",
     "    void Resize(size_t newCapacity) RAPIDJSON_NOEXCEPT {"),
])
patch("allocators.h", [
    ("    bool AddChunk(size_t capacity) {",
     "    bool AddChunk(size_t capacity) RAPIDJSON_NOEXCEPT {"),
])
patch("encodings.h", [
    ("""    enum { supportUnicode = 1 };

    template<typename OutputStream>
    static void Encode(OutputStream& os, unsigned codepoint) {
        if (codepoint <= 0x7F) """,
     """    enum { supportUnicode = 1 };

    template<typename OutputStream>
    static void Encode(OutputStream& os, unsigned codepoint) RAPIDJSON_NOEXCEPT {
        if (codepoint <= 0x7F) """),
])
# The Document handler (GenericDocument's "Implementation of Handler").
handler = []
for sig in ["bool Null()", "bool Bool(bool b)", "bool Int(int i)",
            "bool Uint(unsigned i)", "bool Int64(int64_t i)",
            "bool Uint64(uint64_t i)", "bool Double(double d)",
            "bool StartObject()", "bool StartArray()",
            "bool Key(const Ch* str, SizeType length, bool copy)",
            "bool EndObject(SizeType memberCount)",
            "bool EndArray(SizeType elementCount)"]:
    handler.append(("    " + sig + " {", "    " + sig + " RAPIDJSON_NOEXCEPT {"))
for sig in ["bool RawNumber(const Ch* str, SizeType length, bool copy)",
            "bool String(const Ch* str, SizeType length, bool copy)"]:
    handler.append(("    " + sig + " { \n", "    " + sig + " RAPIDJSON_NOEXCEPT {\n"))
patch("document.h", handler)
