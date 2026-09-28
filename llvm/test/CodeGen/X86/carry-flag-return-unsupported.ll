; RUN: not llc < %s -mtriple=x86_64-pc-windows-msvc -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIN64
; RUN: not llc < %s -mtriple=x86_64-unknown-linux-gnu -o /dev/null 2>&1 | FileCheck %s --check-prefix=ZERO

; Returning in the carry flag is not supported when the epilogue may clobber
; EFLAGS: on Win64, which deallocates the stack with ADD without a frame
; pointer, and with zero-call-used-regs, which clears registers with XOR.

declare void @g()

; WIN64: error: {{.*}} in function win64 {{.*}}: carry flag return on Win64
define { i64, i64, i1 } @win64(i1 %f) #0 {
  call void @g()
  %r = insertvalue { i64, i64, i1 } poison, i1 %f, 2
  ret { i64, i64, i1 } %r
}

; ZERO: error: {{.*}} in function zero {{.*}}: carry flag return with zero-call-used-regs
define { i64, i64, i1 } @zero(i1 %f) #1 {
  %r = insertvalue { i64, i64, i1 } poison, i1 %f, 2
  ret { i64, i64, i1 } %r
}

attributes #0 = { nounwind "x86-carry-flag-return" }
attributes #1 = { nounwind "x86-carry-flag-return" "zero-call-used-regs"="used-gpr" }
