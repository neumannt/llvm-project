// RapidJSON benchmark for the P0709 experiment (see README.md).
//
// Usage: bench <data dir>
//        bench <data dir> repeat <file> <dom|sax> <iterations>
//
// The second form only parses one file repeatedly, e.g. to count retired
// instructions with perf stat.
//
// a) Happy path: parses canada.json, citm_catalog.json and twitter.json
//    into a DOM (Document::Parse) and with a SAX handler that only counts
//    events; reports MB/s.
// b) Errors: parses a corpus of small documents (the tweets of twitter.json
//    and the events of citm_catalog.json, one document each), of which a
//    given fraction is corrupted at a random position; reports ns/document.
//
// Output lines are "result,<section>,<test>,<value>" plus a checksum that
// must be the same for all variants.
#include "rapidjson/document.h"
#include "rapidjson/reader.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace rapidjson;

static unsigned long Checksum = 0;
static void mix(unsigned long V) { Checksum = Checksum * 1000003 + V; }

static std::string readFile(const std::string &Path) {
  std::ifstream In(Path, std::ios::binary);
  if (!In) {
    fprintf(stderr, "cannot read %s\n", Path.c_str());
    exit(1);
  }
  std::stringstream SS;
  SS << In.rdbuf();
  return SS.str();
}

// Best time per call of F over several batches of at least MinBatch seconds.
template <class F> static double bestSeconds(F Fn, double MinBatch = 0.05) {
  using Clock = std::chrono::steady_clock;
  double Best = 1e30;
  for (int Batch = 0; Batch < 10; ++Batch) {
    long N = 0;
    auto T0 = Clock::now();
    double Elapsed;
    do {
      Fn();
      ++N;
      Elapsed = std::chrono::duration<double>(Clock::now() - T0).count();
    } while (Elapsed < MinBatch);
    Best = std::min(Best, Elapsed / N);
  }
  return Best;
}

// Counts SAX events (and a little of their contents). Like the Document
// handler, it does not throw (see patch-rapidjson.py).
struct CountingHandler : BaseReaderHandler<UTF8<>, CountingHandler> {
  unsigned long Events = 0, Chars = 0;
  bool Default() noexcept {
    ++Events;
    return true;
  }
  bool String(const char *, SizeType Length, bool) noexcept {
    ++Events;
    Chars += Length;
    return true;
  }
  bool Key(const char *, SizeType Length, bool) noexcept {
    ++Events;
    Chars += Length;
    return true;
  }
};

static volatile unsigned long Sink;

static unsigned long parseDOM(const std::string &Json) {
  Document D;
  D.Parse(Json.c_str(), Json.size());
  if (D.HasParseError())
    return 1000000 + D.GetParseError() * 100000 + D.GetErrorOffset();
  return D.IsObject() ? D.MemberCount() : D.Size();
}

static unsigned long parseSAX(const std::string &Json) {
  CountingHandler H;
  Reader R;
  StringStream S(Json.c_str());
  ParseResult Res = R.Parse(S, H);
  return Res.IsError() ? 1 : H.Events * 31 + H.Chars;
}

static std::string toString(const Value &V) {
  StringBuffer Buf;
  Writer<StringBuffer> W(Buf);
  V.Accept(W);
  return Buf.GetString();
}

int main(int argc, char **argv) {
  std::string Dir = argc > 1 ? argv[1] : ".";
  if (argc == 6 && std::string(argv[2]) == "repeat") {
    std::string Json = readFile(Dir + "/" + argv[3]);
    bool DOM = std::string(argv[4]) == "dom";
    unsigned long S = 0;
    for (long I = 0, N = atol(argv[5]); I < N; ++I)
      S += DOM ? parseDOM(Json) : parseSAX(Json);
    Sink = S;
    return 0;
  }
  const char *Files[] = {"canada.json", "citm_catalog.json", "twitter.json"};

  // a) Happy path.
  for (const char *File : Files) {
    std::string Json = readFile(Dir + "/" + File);
    double MB = Json.size() / 1e6;
    mix(parseDOM(Json));
    mix(parseSAX(Json));
    printf("result,dom,%s,%.1f\n", File,
           MB / bestSeconds([&] { Sink = parseDOM(Json); }));
    printf("result,sax,%s,%.1f\n", File,
           MB / bestSeconds([&] { Sink = parseSAX(Json); }));
    fflush(stdout);
  }

  // b) Errors: a corpus of small documents.
  std::vector<std::string> Corpus;
  {
    Document Twitter, Citm;
    Twitter.Parse(readFile(Dir + "/twitter.json").c_str());
    Citm.Parse(readFile(Dir + "/citm_catalog.json").c_str());
    for (const Value &Status : Twitter["statuses"].GetArray())
      Corpus.push_back(toString(Status));
    for (const auto &Event : Citm["events"].GetObject())
      Corpus.push_back(toString(Event.value));
  }
  size_t TotalBytes = 0;
  for (const std::string &Doc : Corpus)
    TotalBytes += Doc.size();
  printf("# error corpus: %zu documents, %zu bytes on average\n",
         Corpus.size(), TotalBytes / Corpus.size());

  for (int Percent : {0, 1, 10, 50, 100}) {
    // Corrupt a random position of the chosen documents with a control
    // character, which is invalid everywhere in JSON.
    std::mt19937 Rng(42);
    std::vector<std::string> Docs = Corpus;
    unsigned Failures = 0;
    for (std::string &Doc : Docs) {
      if (std::uniform_int_distribution<int>(0, 99)(Rng) >= Percent)
        continue;
      Doc[std::uniform_int_distribution<size_t>(0, Doc.size() - 1)(Rng)] =
          '\x01';
      ++Failures;
    }
    unsigned long Sum = 0;
    unsigned Failed = 0;
    for (const std::string &Doc : Docs) {
      unsigned long R = parseDOM(Doc);
      Failed += R >= 1000000;
      Sum += R;
    }
    if (Failed != Failures) {
      fprintf(stderr, "corruption did not cause an error\n");
      return 1;
    }
    mix(Sum);
    double Seconds = bestSeconds([&] {
      unsigned long S = 0;
      for (const std::string &Doc : Docs)
        S += parseDOM(Doc);
      Sink = S;
    });
    printf("result,errors,%d%%,%.1f\n", Percent, Seconds / Docs.size() * 1e9);
    fflush(stdout);
  }
  printf("checksum,%lx\n", Checksum);
}
