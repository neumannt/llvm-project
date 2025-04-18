source_group("Header Files" REGULAR_EXPRESSION
  .*[.]h
)
source_group("Header Files\\COFF" REGULAR_EXPRESSION
  COFF/.*[.]h
)
source_group("Header Files\\ELF" REGULAR_EXPRESSION
  ELF/.*[.]h
)
source_group("Header Files\\MachO" REGULAR_EXPRESSION
  MachO/.*[.]h
)
source_group("Header Files\\wasm" REGULAR_EXPRESSION
  wasm/.*[.]h
)
source_group("Source Files" REGULAR_EXPRESSION
  .*[.]cpp
)
source_group("Source Files\\COFF" REGULAR_EXPRESSION  
  COFF/.*[.]cpp
)
source_group("Source Files\\ELF" REGULAR_EXPRESSION
  ELF/.*[.]cpp
)
source_group("Source Files\\MachO" REGULAR_EXPRESSION
  MachO/.*[.]cpp
)
source_group("Source Files\\wasm" REGULAR_EXPRESSION
  wasm/.*[.]cpp
)

add_llvm_component_library(LLVMObjCopy
  Archive.cpp
  ObjCopy.cpp
  ConfigManager.cpp
  COFF/COFFObjcopy.cpp
  COFF/COFFObject.cpp
  COFF/COFFReader.cpp
  COFF/COFFWriter.cpp
  ELF/ELFObjcopy.cpp
  ELF/ELFObject.cpp
  MachO/MachOObjcopy.cpp
  MachO/MachOReader.cpp
  MachO/MachOWriter.cpp
  MachO/MachOLayoutBuilder.cpp
  MachO/MachOObject.cpp
  wasm/WasmObject.cpp
  wasm/WasmReader.cpp
  wasm/WasmWriter.cpp
  wasm/WasmObjcopy.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ObjCopy
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ObjCopy/COFF
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ObjCopy/ELF
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ObjCopy/MachO
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ObjCopy/wasm
  COFF
  ELF
  MachO
  wasm
 
  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  BinaryFormat
  Object
  Support
  MC
  )
