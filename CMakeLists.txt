cmake_minimum_required(VERSION 3.26...4.2)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Off for distributed wheels: they must run on any CPU of their platform.
option(FTS_NATIVE_ARCH "Optimise for the CPU of the build machine (local builds only)" OFF)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_native NB_STATIC
  cpp/src/bindings.cpp
  cpp/src/curves/pillar_curve.cpp
  cpp/src/curves/fits.cpp
  cpp/src/bind_swaps.cpp
  cpp/src/bind_fx.cpp
  cpp/src/bind_market.cpp
  cpp/src/swaps/swap_leg.cpp)
target_include_directories(_native PRIVATE cpp/src)

# Results must match the Python implementation: IEEE arithmetic as written, no
# fast-math reassociation and no fused multiply-add contraction.
if(MSVC)
  target_compile_options(_native PRIVATE /O2 /fp:precise /W4 /permissive-)
  if(FTS_NATIVE_ARCH)
    target_compile_options(_native PRIVATE /arch:AVX2)
  endif()
else()
  target_compile_options(_native PRIVATE -O3 -ffp-contract=off -Wall -Wextra)
  if(FTS_NATIVE_ARCH)
    target_compile_options(_native PRIVATE -march=native)
  endif()
endif()

install(TARGETS _native LIBRARY DESTINATION fintoolsom)
