# -----------------------------------------------------------------------------
# unit_tests/NBody/CMakeLists.txt
#
# Sanity tests for the vendored cstone + ryoanji codebases.
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

# SphexaParticleContainer sanity test: construct, create, sync, verify
# ID-position lockstep round-trip on a single rank.
# Notes:
#   - CUDA_SEPARABLE_COMPILATION is NOT enabled on this test (same reason
#     as ippl_nbody — including <Kokkos_Core.hpp> via SphexaParticleContainer.hpp
#     under RDC mode trips desul's atomics check).
#   - The test provides its own main() that calls ippl::initialize / ippl::finalize
#     because cstone's Domain calls MPI collectives unconditionally — the stock
#     gtest main does not initialize MPI. Hence: no USE_GTEST_MAIN.
add_ippl_test(SphexaContainerSync
  SOURCES SphexaContainerSync.cu
  NO_MPI
  LABELS nbody gpu)
target_link_libraries(SphexaContainerSync PRIVATE
  ippl::nbody)
set_target_properties(SphexaContainerSync PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# BH solver sanity test: BH ≈ direct N² on a small open-BC system.
# Same RDC / main / linkage caveats as SphexaContainerSync above.
# ippl::ryoanji is needed for the GPU directSum() reference — the explicit
# float/double instantiations live in multipole_holder.cu, part of the ryoanji
# glue lib (see RyoanjiDirect for the precedent linkage).
add_ippl_test(SphexaBHSolverDirect
  SOURCES SphexaBHSolverDirect.cu
  NO_MPI
  LABELS nbody gpu)
target_link_libraries(SphexaBHSolverDirect PRIVATE
  ippl::nbody
  ippl::ryoanji)
set_target_properties(SphexaBHSolverDirect PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# Leapfrog kernel sanity test: drift + half-kick correctness in isolation
# (no BH solver). Same RDC / main / linkage pattern as SphexaContainerSync.
add_ippl_test(LeapfrogStepperBasics
  SOURCES LeapfrogStepperBasics.cu
  NO_MPI
  LABELS nbody gpu)
target_link_libraries(LeapfrogStepperBasics PRIVATE
  ippl::nbody)
set_target_properties(LeapfrogStepperBasics PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# Periodic-wrap kernel test: covers all-periodic, all-open, and mixed BCs.
add_ippl_test(PeriodicWrapKernel
  SOURCES PeriodicWrapKernel.cu
  NO_MPI
  LABELS nbody gpu)
target_link_libraries(PeriodicWrapKernel PRIVATE
  ippl::nbody)
set_target_properties(PeriodicWrapKernel PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
