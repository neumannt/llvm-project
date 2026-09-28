// RUN: %clang -### --target=x86_64-linux-gnu -fstatic-exceptions -fstatic-exceptions-abi=carry %s 2>&1 | FileCheck %s --check-prefix=CARRY
// RUN: %clang -### --target=aarch64-linux-gnu -fstatic-exceptions -fstatic-exceptions-abi=register %s 2>&1 | FileCheck %s --check-prefix=REG
// RUN: not %clang -### --target=aarch64-linux-gnu -fstatic-exceptions -fstatic-exceptions-abi=carry %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: not %clang -### --target=x86_64-pc-windows-msvc -fstatic-exceptions -fstatic-exceptions-abi=carry %s 2>&1 | FileCheck %s --check-prefix=ERR-WIN64
// RUN: not %clang -### --target=x86_64-w64-mingw32 -fstatic-exceptions -fstatic-exceptions-abi=carry %s 2>&1 | FileCheck %s --check-prefix=ERR-MINGW
// RUN: %clang -### --target=i686-pc-windows-msvc -fstatic-exceptions -fstatic-exceptions-abi=carry %s 2>&1 | FileCheck %s --check-prefix=CARRY
// RUN: not %clang -### --target=x86_64-linux-gnu -fstatic-exceptions -fstatic-exceptions-abi=carry -fzero-call-used-regs=used-gpr %s 2>&1 | FileCheck %s --check-prefix=ERR-ZERO
// RUN: %clang -### --target=x86_64-linux-gnu -fstatic-exceptions -fstatic-exceptions-abi=carry -fzero-call-used-regs=skip %s 2>&1 | FileCheck %s --check-prefix=CARRY

// CARRY: "-cc1"{{.*}} "-fstatic-exceptions-abi=carry"
// REG: "-cc1"{{.*}} "-fstatic-exceptions-abi=register"
// ERR: error: unsupported option '-fstatic-exceptions-abi=carry' for target 'aarch64-unknown-linux-gnu'
// ERR-WIN64: error: unsupported option '-fstatic-exceptions-abi=carry' for target 'x86_64-pc-windows-msvc'
// ERR-ZERO: error: invalid argument '-fstatic-exceptions-abi=carry' not allowed with '-fzero-call-used-regs=used-gpr'
// ERR-MINGW: error: unsupported option '-fstatic-exceptions-abi=carry' for target 'x86_64-w64-windows-gnu'
