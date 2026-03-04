cmake_minimum_required(VERSION 3.18)
project(cathalTest LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
include(CheckCXXCompilerFlag)

# Source
add_library(cathalTest MODULE src/functionDefs.cpp src/myPBmodule.cpp)
# or: pybind11_add_module(cathalTest src/functionDefs.cpp src/myPBmodule.cpp)

set_target_properties(cathalTest PROPERTIES
  CXX_STANDARD 17
  PREFIX ""                 # no "lib" prefix
# SUFFIX ".pyd"             # <-- critical on Windows
  LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${PROJECT_SOURCE_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}"
)

target_compile_features(cathalTest PUBLIC cxx_std_17)

target_compile_options(cathalTest PRIVATE
    -O0
#    -march=native
#    -ffast-math
)

target_compile_definitions(cathalTest PRIVATE NDEBUG)

if(APPLE)
    # 1. Tell the compiler to recognize the pragmas
    target_compile_options(cathalTest PRIVATE -Xpreprocessor -fopenmp)
    
    # 2. DO NOT use target_link_libraries(cathalTest PRIVATE "-lomp")
    # Instead, we just provide the header path so it compiles.
    target_include_directories(cathalTest PRIVATE /opt/homebrew/opt/llvm/include)
    
    # 3. Link with 'undefined dynamic_lookup' 
    # This is a macOS-specific linker flag that says "find symbols at runtime"
    target_link_options(cathalTest PRIVATE "-Wl,-undefined,dynamic_lookup")
    target_link_libraries(cathalTest PRIVATE pybind11::module)
# endif()
else()
   target_link_libraries(cathalTest PRIVATE pybind11::module OpenMP::OpenMP_CXX)
   
endif()
# target_link_libraries(cathalTest PRIVATE pybind11::module OpenMP::OpenMP_CXX)


target_include_directories(cathalTest PRIVATE ${PROJECT_SOURCE_DIR}/include)


# MSVC-specific tuning
if(MSVC)
  # Prefer the newer LLVM OpenMP switch when available
  check_cxx_compiler_flag("/openmp:llvm" HAS_OPENMP_LLVM)
  if(HAS_OPENMP_LLVM)
    target_compile_options(cathalTest PRIVATE /openmp:llvm)
  else()
    target_compile_options(cathalTest PRIVATE /openmp)
  endif()

  # Release: optimize hard, enable LTCG; Debug: symbols, no optimization
  target_compile_options(cathalTest PRIVATE
    $<$<CONFIG:Release>:/O2 /GL /fp:fast /arch:AVX2 /DNDEBUG>
    $<$<CONFIG:Debug>:/Zi /Od /D_DEBUG /DWIN32_LEAN_AND_MEAN /D_CRT_SECURE_NO_WARNINGS>
  )
  target_link_options(cathalTest PRIVATE
    $<$<CONFIG:Release>:/LTCG>
  )
  # Optional: codegen summary in Release to sanity-check vectorization
  # target_compile_options(cathalTest PRIVATE $<$<CONFIG:Release>:/d2cgsummary>)
else()
  # GCC/Clang (Linux/macOS)
  target_compile_options(cathalTest PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -ffast-math -fno-math-errno -fno-trapping-math -DNDEBUG>
    $<$<CONFIG:Debug>:-O0 -g>
  )
endif()
