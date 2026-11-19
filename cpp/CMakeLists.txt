cmake_minimum_required(VERSION 3.18)
project(neurovol_iv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # plain -std=c++17, no GNU extensions

option(NEUROVOL_BUILD_PYTHON "Build the pybind11 extension module" ON)
option(NEUROVOL_WERROR "Treat compiler warnings as errors" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Warnings and optimization. CMake's default Release flag is -O3; this project
# uses -O2 on purpose (see docs/INTERVIEW_PREP.md). No -ffast-math: it lets the
# compiler assume NaN never happens, and the solver uses NaN as its error value.
if(MSVC)
  set(NEUROVOL_WARNINGS /W4 /permissive-)
  if(NEUROVOL_WERROR)
    list(APPEND NEUROVOL_WARNINGS /WX)
  endif()
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
else()
  set(NEUROVOL_WARNINGS -Wall -Wextra -Wpedantic)
  if(NEUROVOL_WERROR)
    list(APPEND NEUROVOL_WARNINGS -Werror)
  endif()
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()

# On MinGW (llvm-mingw / MSYS2), link the C++ runtime statically so the test
# executable and the .pyd do not depend on libc++.dll / libstdc++-6.dll being
# on PATH. Python 3.8+ does not search PATH for extension dependencies.
if(MINGW)
  set(NEUROVOL_STATIC_RUNTIME -static)
endif()

# --- Core library: no Python, no exceptions thrown, no heap ---------------------
add_library(iv_core STATIC src/iv_core.cpp)
target_include_directories(iv_core PUBLIC include)
target_compile_options(iv_core PRIVATE ${NEUROVOL_WARNINGS})
set_target_properties(iv_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- C++ unit tests -----------------------------------------------------------
enable_testing()
add_executable(test_iv_core tests/test_iv_core.cpp)
target_link_libraries(test_iv_core PRIVATE iv_core)
target_compile_options(test_iv_core PRIVATE ${NEUROVOL_WARNINGS})
target_link_options(test_iv_core PRIVATE ${NEUROVOL_STATIC_RUNTIME})
add_test(NAME test_iv_core COMMAND test_iv_core)

# --- Python extension ---------------------------------------------------------
if(NEUROVOL_BUILD_PYTHON)
  find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
  if(NOT pybind11_DIR)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE pybind11_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_iv_core src/bindings.cpp)
  target_link_libraries(_iv_core PRIVATE iv_core)
  target_compile_options(_iv_core PRIVATE ${NEUROVOL_WARNINGS})
  target_link_options(_iv_core PRIVATE ${NEUROVOL_STATIC_RUNTIME})

  # Record the flags in the module so bench/benchmark.py can report what the
  # loaded binary was actually built with.
  string(REPLACE ";" " " _warn_str "${NEUROVOL_WARNINGS}")
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
  target_compile_definitions(_iv_core PRIVATE
    NEUROVOL_BUILD_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}} -std=c++17 ${_warn_str}"
    NEUROVOL_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

  # Put the built module next to iv_solver.py so `from . import _iv_core` works.
  set(NEUROVOL_PKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vol_surface_engine")
  set_target_properties(_iv_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${NEUROVOL_PKG_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${NEUROVOL_PKG_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${NEUROVOL_PKG_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${NEUROVOL_PKG_DIR}")
endif()
