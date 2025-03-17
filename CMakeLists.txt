# ------------------------------------------------------------------------------
# CMake version requirement
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20.3)

# ------------------------------------------------------------------------------
# Policies - use the latest of everything
# ------------------------------------------------------------------------------
cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})

# ------------------------------------------------------------------------------
# Project declaration and version
# ------------------------------------------------------------------------------
project(
  IPPL
  VERSION 3.2
  LANGUAGES CXX)
set(IPPL_VERSION_NAME "V${IPPL_VERSION_MAJOR}.${IPPL_VERSION_MINOR}")
message(
  STATUS
    "Configuring IPPL version ${IPPL_VERSION_MAJOR}.${IPPL_VERSION_MINOR} : ${IPPL_VERSION_NAME}"
)

# ------------------------------------------------------------------------------
# CMake configuration
# ------------------------------------------------------------------------------
set(CMAKE_MODULE_PATH
    "${CMAKE_MODULE_PATH}"
    "${PROJECT_SOURCE_DIR}/cmake;${PROJECT_SOURCE_DIR}/CMakeModules")

# ------------------------------------------------------------------------------
# cmake modules
# ------------------------------------------------------------------------------
include(FetchContent)
include(Messages)
include(ExternalProject)

option(DOWNLOAD_EXTRACT_TIMESTAMP ON FORCE)

# ------------------------------------------------------------------------------
# Macros
# ------------------------------------------------------------------------------
macro(remove_flag_from_target _target _flag)
  get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
  if(_target_cxx_flags)
    list(REMOVE_ITEM _target_cxx_flags ${_flag})
    set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS
                                                "${_target_cxx_flags}")
  endif()
endmacro()

# ------------------------------------------------------------------------------
# BUILD_TYPE
# ------------------------------------------------------------------------------
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      RelWithDebInfo
      CACHE
        STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
  colour_message("${Yellow}"
      "CMAKE_BUILD_TYPE not set, defaulting to ${CMAKE_BUILD_TYPE}"
  )
else()
    colour_message("${Green}" "Build type is: ${CMAKE_BUILD_TYPE}")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
  add_compile_definitions(NOPAssert)
endif()

# ------------------------------------------------------------------------------
# CXX settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINGO} -O3 -g
# ") set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3") set
# (CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# ------------------------------------------------------------------------------
# User options
# ------------------------------------------------------------------------------
option(USE_STATIC_LIBRARIES "Link with static libraries if available" ON)
option(ENABLE_ALPINE "Enable Alpine" OFF)
option(ENABLE_COSMOLOGY "Enable Cosmology" OFF)
option(ENABLE_SOLVERS "Enable IPPL solvers" ON)
option(ENABLE_TESTS "Enable IPPL tests" OFF)
option(ENABLE_UNIT_TESTS "Enable unit tests" OFF)

# ------------------------------------------------------------------------------
# Backend platform selection
# ------------------------------------------------------------------------------
if(NOT IPPL_PLATFORMS)
colour_message("${Yellow}"
      "IPPL_PLATFORMS flag is not set, defaulting to serial backend"
  )
  set(IPPL_PLATFORMS
      "SERIAL"
      CACHE STRING "Select the backend platform for IPPL")
  set_property(CACHE IPPL_PLATFORMS PROPERTY STRINGS CUDA OPENMP SERIAL)
else()
colour_message("${Green}"
      "IPPL_PLATFORMS flag is set to: ${IPPL_PLATFORMS}")
endif()

if(Ippl_PLATFORMS)
  message(FATAL_ERROR "Ippl_PLATFORMS is deprecated, use IPPL_PLATFORMS")
endif()

string(TOUPPER "${IPPL_PLATFORMS}" IPPL_PLATFORMS)
# Perform case distinction
if("${IPPL_PLATFORMS}" STREQUAL "CUDA")
colour_message("${Green}" "CUDA backend enabled")
  set(Kokkos_ENABLE_CUDA
      ON
      CACHE BOOL "Enable Kokkos CUDA Backend")
  set(Heffte_ENABLE_CUDA
      ON
      CACHE BOOL "Set Heffte CUDA Backend")

elseif("${IPPL_PLATFORMS}" STREQUAL "OPENMP")
colour_message("${Green}" "OpenMP backend enabled")
  if(NOT Heffte_ENABLE_FFTW)
    colour_message("${Yellow}"
      "Because Heffte_ENABLE_FFTW is not set defaulting to Stock with AVX"
    )
    set(Heffte_ENABLE_AVX2
        ON
        CACHE BOOL "Enable Heffte Stock backend lol")
  else()
    colour_message("${Green}" "Enabled FFTW")
  endif()
  set(Kokkos_ENABLE_OPENMP
      ON
      CACHE BOOL "Enable Kokkos OpenMP Backend")

elseif("${IPPL_PLATFORMS}" STREQUAL "SERIAL")
  colour_message("${Green}" "Serial backend enabled")
  if(NOT Heffte_ENABLE_FFTW)
    colour_message("${Yellow}"
      "Because Heffte_ENABLE_FFTW is not set defaulting to Stock with AVX"
    )
    set(Heffte_ENABLE_AVX2
        ON
        CACHE BOOL "Enable Heffte Stock backend lol")
  endif()

elseif("${IPPL_PLATFORMS}" STREQUAL "CUDA;OPENMP" OR "${IPPL_PLATFORMS}"
                                                     STREQUAL "OPENMP;CUDA")
  colour_message("${Green}" "CUDA and OpenMP backend enabled")
  set(Heffte_ENABLE_CUDA
      ON
      CACHE BOOL "Set Heffte CUDA Backend")
  set(Kokkos_ENABLE_OPENMP
      ON
      CACHE BOOL "Enable Kokkos OpenMP Backend")
  set(Kokkos_ENABLE_CUDA
      ON
      CACHE BOOL "Enable Kokkos CUDA Backend")

else()
  message(FATAL_ERROR "Unknown or unsupported platform: ${IPPL_PLATFORMS}")
  # Handle unknown or unsupported platform
endif()

# ------------------------------------------------------------------------------
# MPI
# ------------------------------------------------------------------------------
find_package(MPI REQUIRED)

message(STATUS "The C++ compiler identification is: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "The C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "The MPI C++ compiler is: ${MPI_CXX_COMPILER}")
message(STATUS "The underlying C++ compiler is: ${CMAKE_CXX_COMPILER}")

# ------------------------------------------------------------------------------
# setup Kokkos
# ------------------------------------------------------------------------------
if(NOT Kokkos_VERSION)
  set(Kokkos_VERSION "4.1.00")
  colour_message("${Green}" "Selecting default Kokkos version: ${Kokkos_VERSION}")
else()
  colour_message("${Green}" "Selected Kokkos version: ${Kokkos_VERSION}")
endif()

get_external_project(
  PROJECT_NAME Kokkos
  VERSION ${Kokkos_VERSION}
  DEBUG
  DOWNLOAD_EXTRACT_TIMESTAMP ${DOWNLOAD_EXTRACT_TIMESTAMP}
  URL https://github.com/kokkos/kokkos/archive/refs/tags/${Kokkos_VERSION}.tar.gz
)

# ------------------------------------------------------------------------------
# Heffte
# ------------------------------------------------------------------------------
option(ENABLE_FFT "Enable FFT transform (Heffte)" ON)

if(ENABLE_FFT)
  add_definitions(-DENABLE_FFT)

  if(NOT HEFFTE_COMMIT_HASH)
    set(HEFFTE_COMMIT_HASH 9eab7c0eb18e86acaccc2b5699b30e85a9e7bdda)
  endif()

  # if user has specified a version, use that when trying to find the package
  if(Heffte_VERSION)
    if(Heffte_VERSION MATCHES "MASTER")
      set(HEFFTE_MODE
        VERSION 2.2.0
        GIT_REPO "https://github.com/icl-utk-edu/heffte" GIT_TAG ${HEFFTE_COMMIT_HASH})
    else()
      set(HEFFTE_MODE
        VERSION ${Heffte_VERSION}
        DOWNLOAD_EXTRACT_TIMESTAMP True
        URL "https://github.com/icl-utk-edu/heffte/archive/refs/tags/v${Heffte_VERSION}.tar.gz")
    endif()
  else()
    # just get latest master branch
    set(HEFFTE_MODE
        GIT_REPO "https://github.com/icl-utk-edu/heffte" GIT_TAG ${HEFFTE_COMMIT_HASH})
  endif()

  get_external_project(PROJECT_NAME Heffte DEBUG ${HEFFTE_MODE})

  # if fetch content downloaded the project and we are building it in place
  if(${Heffte_POPULATED})
    colour_message("${Green}" "Heffte downloaded using mode: ${HEFFTE_MODE}")
    # Required to suppress warnings related to heffte
    target_include_directories(Heffte SYSTEM
                               PUBLIC "${Heffte_SOURCE_DIR}/include")
    colour_message("${Red}" "Heffte source directory: ${Heffte_SOURCE_DIR}")
    set(Heffte_DIR "${Heffte_SOURCE_DIR}")
    remove_flag_from_target(Heffte "-Werror")
    remove_flag_from_target(Heffte "-Wall")
    remove_flag_from_target(Heffte "-Wextra")
    # make sure locally compiled and imported version use same lib name
    add_library(Heffte::Heffte ALIAS Heffte)
  endif()
endif()

# ------------------------------------------------------------------------------
# Debug mode : Address sanitizer / other flags
# ------------------------------------------------------------------------------
if(NOT "${IPPL_PLATFORMS}" STREQUAL "CUDA" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Use sanitizer flags for GCC or Clang only
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")
endif()

if("${Kokkos_DEVICES}" MATCHES "CUDA")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -G")
endif()

# Suppress erroneous buffer overflow warnings when compiling under GCC 12
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_CXX_COMPILER_VERSION}" MATCHES "12*")
  add_compile_options(-Wno-stringop-overflow)
  add_compile_options(-Wno-array-bounds)
  add_compile_options(-Wno-restrict)
endif()

# ------------------------------------------------------------------------------
# Option settings
# ------------------------------------------------------------------------------
option(
  USE_ALTERNATIVE_VARIANT
  "Use modified variant implementation (required for CUDA 12.2 + GCC 12.3.0)"
  OFF)
if(USE_ALTERNATIVE_VARIANT)
  add_definitions(-DUSE_ALTERNATIVE_VARIANT)
endif()

# ------------------------------------------------------------------------------
# legacy build options
# ------------------------------------------------------------------------------
add_compile_options(-Wall)
add_compile_options(-Wunused)
add_compile_options(-Wextra)
# add_compile_options (-Werror)

# allow deprecated functions
add_compile_options(-Wno-deprecated-declarations)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-Wno-deprecated-copy)
  add_compile_options(-Wno-sign-compare)
endif()

# ------------------------------------------------------------------------------
# subdirs
# ------------------------------------------------------------------------------
add_subdirectory(src)


if(ENABLE_TESTS)
  add_subdirectory(test)
endif()

if(ENABLE_ALPINE)
  if(NOT ENABLE_SOLVERS)
    message(
      FATAL_ERROR
        "Cannot enable Alpine since Solver not enabled (-DENABLE_SOLVERS=ON)!")
  endif()
  if(NOT ENABLE_FFT)
    message(
      FATAL_ERROR
        "Cannot enable Alpine since FFT not enabled (-DENABLE_FFT=ON)!")
  endif()
  message(STATUS "Enable Alpine")
  add_subdirectory(alpine)
endif()

if(ENABLE_COSMOLOGY)
  if(NOT ENABLE_SOLVERS)
    message(
      FATAL_ERROR
        "Cannot enable Cosmology since Solver not enabled (-DENABLE_SOLVERS=ON)!"
    )
  endif()
  if(NOT ENABLE_FFT)
    message(
      FATAL_ERROR
        "Cannot enable Cosmology since FFT not enabled (-DENABLE_FFT=ON)!")
  endif()
  message(STATUS "Enable Cosmology")
  add_subdirectory(cosmology)
endif()

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------
if(ENABLE_UNIT_TESTS)
  if(NOT ENABLE_FFT)
    message(
      WARNING
        "Enabling unit tests even though FFT not enabled (-DENABLE_FFT=ON)!")
  endif()
  include(GoogleTest)

  enable_testing()

  get_external_project(
    PROJECT_NAME GTest
    DEBUG
    GIT_REPO "https://github.com/google/googletest"
    GIT_TAG "v1.16.0")

  if(${GTest_POPULATED})
    # For Windows: Prevent overriding the parent project's compiler/linkersettings
    set(gtest_force_shared_crt
        ON
        CACHE BOOL "" FORCE)
  endif()
  add_subdirectory(unit_tests)
endif()

# ------------------------------------------------------------------------------
# install/configure
# ------------------------------------------------------------------------------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config_install.cmake)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config_install.cmake
  DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/${PROJECT_NAME}"
  RENAME ${PROJECT_NAME}Config.cmake)

# vi: set et ts=4 sw=4 sts=4:

# Local Variables: mode: cmake cmake-tab-width: 4 indent-tabs-mode: nil
# require-final-newline: nil End:
