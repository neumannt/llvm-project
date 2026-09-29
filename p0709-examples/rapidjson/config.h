// Error handling variants of RapidJSON's parser for the P0709 experiment.
// Included (with -include) before RapidJSON; select a variant with
//   -DRJ_CODES    the original: error codes, checked after each call
//   -DRJ_DYNAMIC  traditional C++ exceptions
//   -DRJ_STATIC   P0709 static exceptions ('throws', -fstatic-exceptions)
// The functions of the parser that can fail are declared RAPIDJSON_THROWS:
// noexcept, nothing (may throw) or 'throws', respectively. In all variants,
// the error code and offset are recorded in the reader before the error is
// reported, so Document::Parse returns the same result.
#ifndef RJ_P0709_CONFIG_H
#define RJ_P0709_CONFIG_H

#if defined(RJ_DYNAMIC)

namespace rj_p0709 {
struct ParseAbort {};
} // namespace rj_p0709
#define RAPIDJSON_PARSE_ERROR_EARLY_RETURN(value) ((void)0)
#define RAPIDJSON_PARSE_ERROR_NORETURN(parseErrorCode, offset)                \
  do {                                                                         \
    SetParseError(parseErrorCode, offset);                                     \
    throw ::rj_p0709::ParseAbort();                                            \
  } while (0)
#define RAPIDJSON_P0709_CATCH catch (const ::rj_p0709::ParseAbort &)

#elif defined(RJ_STATIC)

#include <error>
#define RAPIDJSON_THROWS throws
#define RAPIDJSON_PARSE_ERROR_EARLY_RETURN(value) ((void)0)
#define RAPIDJSON_PARSE_ERROR_NORETURN(parseErrorCode, offset)                \
  do {                                                                         \
    SetParseError(parseErrorCode, offset);                                     \
    throw std::errc::illegal_byte_sequence;                                    \
  } while (0)
#define RAPIDJSON_P0709_CATCH catch (std::error)

#elif defined(RJ_CODES)

// Nothing throws in a world of error codes.
#define RAPIDJSON_THROWS noexcept

#else
#error "select a variant: -DRJ_CODES, -DRJ_DYNAMIC or -DRJ_STATIC"
#endif

#endif
