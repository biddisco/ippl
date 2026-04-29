# -----------------------------------------------------------------------------
# unit_tests/NBody/CMakeLists.txt
#
# Phase-0 sanity tests for the vendored cstone + ryoanji codebases.
# Each source file is a verbatim copy of an upstream SPH-EXA unit test.
# These tests verify that the vendored sources compile, link, and behave
# the same way they do in the upstream tree.
#
# Only built when IPPL_ENABLE_NBODY=ON (which already requires CUDA in
# IPPL_PLATFORMS — see top-level CMakeLists.txt).
#
# Note: add_ippl_test() accepts a LINK_LIBS argument but currently only honors
# it on the INTEGRATION branch. We attach the cstone/ryoanji targets via an
# explicit target_link_libraries() call after each add_ippl_test() instead.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding NBody unit tests found in ${_relPath}")

# CPU-only cstone tests: header-only, no GPU code, no MPI.
add_ippl_test(CstoneArray
  NO_MPI
  USE_GTEST_MAIN
  LABELS nbody)
target_link_libraries(CstoneArray PRIVATE ippl::cstone_headers)

add_ippl_test(CstoneBox
  NO_MPI
  USE_GTEST_MAIN
  LABELS nbody)
target_link_libraries(CstoneBox PRIVATE ippl::cstone_headers)

# CPU ryoanji test: exercises multipole kernel headers + cstone headers.
add_ippl_test(RyoanjiKernelCpu
  NO_MPI
  USE_GTEST_MAIN
  LABELS nbody)
target_link_libraries(RyoanjiKernelCpu PRIVATE
  ippl::ryoanji_headers
  ippl::cstone_headers)

# GPU ryoanji test: exercises the direct N² kernel from ryoanji_kernels.
# The .cu source is detected as CUDA automatically because
# enable_language(CUDA) was called at the top level when IPPL_ENABLE_NBODY=ON.
add_ippl_test(RyoanjiDirect
  SOURCES RyoanjiDirect.cu
  NO_MPI
  USE_GTEST_MAIN
  LABELS nbody gpu)
target_link_libraries(RyoanjiDirect PRIVATE
  ippl::ryoanji
  ippl::cstone_gpu)
set_target_properties(RyoanjiDirect PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)
