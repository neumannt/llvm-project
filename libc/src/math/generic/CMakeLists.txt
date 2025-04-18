add_entrypoint_object(
  ceil
  SRCS
    ceil.cpp
  HDRS
    ../ceil.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  ceilf
  SRCS
    ceilf.cpp
  HDRS
    ../ceilf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  ceill
  SRCS
    ceill.cpp
  HDRS
    ../ceill.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_object_library(
  math_utils
  SRCS
    math_utils.cpp
  HDRS
    math_utils.h
  DEPENDS
    libc.include.errno
    libc.include.math
    libc.src.errno.__errno_location
)

add_object_library(
  sincosf_utils
  HDRS
    sincosf_utils.h
  SRCS
    sincosf_data.cpp
  DEPENDS
    .math_utils
)

add_entrypoint_object(
  cosf
  SRCS
    cosf.cpp
  HDRS
    ../cosf.h
  DEPENDS
    .sincosf_utils
    libc.include.math
    libc.src.errno.__errno_location
  COMPILE_OPTIONS
    -O3
)

add_entrypoint_object(
  sinf
  SRCS
    sinf.cpp
  HDRS
    ../sinf.h
  DEPENDS
    .sincosf_utils
    libc.include.math
    libc.src.errno.__errno_location
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O3
)

add_entrypoint_object(
  sincosf
  SRCS
    sincosf.cpp
  HDRS
    ../sincosf.h
  DEPENDS
    .sincosf_utils
    libc.include.math
    libc.src.errno.__errno_location
  COMPILE_OPTIONS
    -O3
)

add_entrypoint_object(
  fabs
  SRCS
    fabs.cpp
  HDRS
    ../fabs.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fabsf
  SRCS
    fabsf.cpp
  HDRS
    ../fabsf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fabsl
  SRCS
    fabsl.cpp
  HDRS
    ../fabsl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  trunc
  SRCS
    trunc.cpp
  HDRS
    ../trunc.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  truncf
  SRCS
    truncf.cpp
  HDRS
    ../truncf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  truncl
  SRCS
    truncl.cpp
  HDRS
    ../truncl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  floor
  SRCS
    floor.cpp
  HDRS
    ../floor.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  floorf
  SRCS
    floorf.cpp
  HDRS
    ../floorf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  floorl
  SRCS
    floorl.cpp
  HDRS
    ../floorl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  round
  SRCS
    round.cpp
  HDRS
    ../round.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  roundf
  SRCS
    roundf.cpp
  HDRS
    ../roundf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  roundl
  SRCS
    roundl.cpp
  HDRS
    ../roundl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  lround
  SRCS
    lround.cpp
  HDRS
    ../lround.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  lroundf
  SRCS
    lroundf.cpp
  HDRS
    ../lroundf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  lroundl
  SRCS
    lroundl.cpp
  HDRS
    ../lroundl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  llround
  SRCS
    llround.cpp
  HDRS
    ../llround.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  llroundf
  SRCS
    llroundf.cpp
  HDRS
    ../llroundf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  llroundl
  SRCS
    llroundl.cpp
  HDRS
    ../llroundl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  rint
  SRCS
    rint.cpp
  HDRS
    ../rint.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  rintf
  SRCS
    rintf.cpp
  HDRS
    ../rintf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  rintl
  SRCS
    rintl.cpp
  HDRS
    ../rintl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  lrint
  SRCS
    lrint.cpp
  HDRS
    ../lrint.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  lrintf
  SRCS
    lrintf.cpp
  HDRS
    ../lrintf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  lrintl
  SRCS
    lrintl.cpp
  HDRS
    ../lrintl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  llrint
  SRCS
    llrint.cpp
  HDRS
    ../llrint.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  llrintf
  SRCS
    llrintf.cpp
  HDRS
    ../llrintf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  llrintl
  SRCS
    llrintl.cpp
  HDRS
    ../llrintl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  nearbyint
  SRCS
    nearbyint.cpp
  HDRS
    ../nearbyint.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  nearbyintf
  SRCS
    nearbyintf.cpp
  HDRS
    ../nearbyintf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  nearbyintl
  SRCS
    nearbyintl.cpp
  HDRS
    ../nearbyintl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_object_library(
  exp_utils
  HDRS
    exp_utils.h
  SRCS
    exp_utils.cpp
  DEPENDS
    .math_utils
)

add_entrypoint_object(
  expf
  SRCS
    expf.cpp
  HDRS
    ../expf.h
  DEPENDS
    .exp_utils
    .math_utils
    libc.include.math
)

add_entrypoint_object(
  exp2f
  SRCS
    exp2f.cpp
  HDRS
    ../exp2f.h
  DEPENDS
    .exp_utils
    .math_utils
    libc.include.math
)

add_entrypoint_object(
  expm1f
  SRCS
    expm1f.cpp
  HDRS
    ../expm1f.h
  DEPENDS
    libc.include.math
    libc.src.math.expf
    libc.src.math.fabsf
)

add_entrypoint_object(
  copysign
  SRCS
    copysign.cpp
  HDRS
    ../copysign.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  copysignf
  SRCS
    copysignf.cpp
  HDRS
    ../copysignf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  copysignl
  SRCS
    copysignl.cpp
  HDRS
    ../copysignl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  frexp
  SRCS
    frexp.cpp
  HDRS
    ../frexp.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  frexpf
  SRCS
    frexpf.cpp
  HDRS
    ../frexpf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  frexpl
  SRCS
    frexpl.cpp
  HDRS
    ../frexpl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  ilogb
  SRCS
    ilogb.cpp
  HDRS
    ../ilogb.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  ilogbf
  SRCS
    ilogbf.cpp
  HDRS
    ../ilogbf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  ilogbl
  SRCS
    ilogbl.cpp
  HDRS
    ../ilogbl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  ldexp
  SRCS
    ldexp.cpp
  HDRS
    ../ldexp.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  ldexpf
  SRCS
    ldexpf.cpp
  HDRS
    ../ldexpf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  ldexpl
  SRCS
    ldexpl.cpp
  HDRS
    ../ldexpl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_object_library(
  common_constants
  HDRS
    common_constants.h
  SRCS
    common_constants.cpp
)

add_entrypoint_object(
  log10f
  SRCS
    log10f.cpp
  HDRS
    ../log10f.h
  DEPENDS
    .common_constants
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O3
)

add_entrypoint_object(
  log1pf
  SRCS
    log1pf.cpp
  HDRS
    ../log1pf.h
  DEPENDS
    .common_constants
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O3
)

add_entrypoint_object(
  log2f
  SRCS
    log2f.cpp
  HDRS
    ../log2f.h
  DEPENDS
    .common_constants
    libc.src.__support.FPUtil.fputil
    COMPILE_OPTIONS
    -O3
)

add_entrypoint_object(
  logf
  SRCS
    logf.cpp
  HDRS
    ../logf.h
  DEPENDS
    .common_constants
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O3
)

add_entrypoint_object(
  logb
  SRCS
    logb.cpp
  HDRS
    ../logb.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  logbf
  SRCS
    logbf.cpp
  HDRS
    ../logbf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  logbl
  SRCS
    logbl.cpp
  HDRS
    ../logbl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  modf
  SRCS
    modf.cpp
  HDRS
    ../modf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  modff
  SRCS
    modff.cpp
  HDRS
    ../modff.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  modfl
  SRCS
    modfl.cpp
  HDRS
    ../modfl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fmin
  SRCS
    fmin.cpp
  HDRS
    ../fmin.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fminf
  SRCS
    fminf.cpp
  HDRS
    ../fminf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fminl
  SRCS
    fminl.cpp
  HDRS
    ../fminl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fmax
  SRCS
    fmax.cpp
  HDRS
    ../fmax.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fmaxf
  SRCS
    fmaxf.cpp
  HDRS
    ../fmaxf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fmaxl
  SRCS
    fmaxl.cpp
  HDRS
    ../fmaxl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  sqrt
  SRCS
    sqrt.cpp
  HDRS
    ../sqrt.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
    libc.src.__support.FPUtil.sqrt
  COMPILE_OPTIONS
    -O3
)

add_entrypoint_object(
  sqrtf
  SRCS
    sqrtf.cpp
  HDRS
    ../sqrtf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
    libc.src.__support.FPUtil.sqrt
  COMPILE_OPTIONS
    -O3
)

add_entrypoint_object(
  sqrtl
  SRCS
    sqrtl.cpp
  HDRS
    ../sqrtl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
    libc.src.__support.FPUtil.sqrt
  COMPILE_OPTIONS
    -O3
)

add_entrypoint_object(
  remquof
  SRCS
    remquof.cpp
  HDRS
    ../remquof.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  remquo
  SRCS
    remquo.cpp
  HDRS
    ../remquo.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  remquol
  SRCS
    remquol.cpp
  HDRS
    ../remquol.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  remainderf
  SRCS
    remainderf.cpp
  HDRS
    ../remainderf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  remainder
  SRCS
    remainder.cpp
  HDRS
    ../remainder.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  remainderl
  SRCS
    remainderl.cpp
  HDRS
    ../remainderl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  hypotf
  SRCS
    hypotf.cpp
  HDRS
    ../hypotf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
    libc.src.__support.FPUtil.sqrt
  COMPILE_OPTIONS
    -O3
)

add_entrypoint_object(
  fdim
  SRCS
    fdim.cpp
  HDRS
    ../fdim.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fdimf
  SRCS
    fdimf.cpp
  HDRS
    ../fdimf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fdiml
  SRCS
    fdiml.cpp
  HDRS
    ../fdiml.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  hypot
  SRCS
    hypot.cpp
  HDRS
    ../hypot.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O3
)

add_entrypoint_object(
  nextafter
  SRCS
    nextafter.cpp
  HDRS
    ../nextafter.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  nextafterf
  SRCS
    nextafterf.cpp
  HDRS
    ../nextafterf.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  nextafterl
  SRCS
    nextafterl.cpp
  HDRS
    ../nextafterl.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
    -O2
)

add_object_library(
  dp_trig
  SRCS
    dp_trig.cpp
  HDRS
    dp_trig.h
  DEPENDS
    libc.src.__support.FPUtil.fputil
  COMPILE_OPTIONS
   -O3
)
