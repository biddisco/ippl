# -----------------------------------------------------------------------------
# extern/cstone/CMakeLists.txt
#
# Vendored copy of the Cornerstone Octree library (cstone) from SPH-EXA,
# providing distributed octree construction, SFC sorting, halo exchange and
# GPU primitives consumed by Ryoanji's Barnes-Hut solver.
#
# Upstream: https://github.com/sphexa-org/SPH-EXA  (domain/include/cstone/)
# License:  MIT (see LICENSE in the upstream repository)
#
# This CMake file is local to the IPPL fork and replaces the upstream
# domain/CMakeLists.txt + per-directory CMakeLists.txt files.
# -----------------------------------------------------------------------------

# cstone host paths are decorated with OpenMP pragmas. When OpenMP is
# unavailable we still want the pragmas to be honored as no-ops without
# producing -Wunknown-pragmas warnings, so make OpenMP a hard requirement
# (matching upstream domain/CMakeLists.txt).
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Header-only target: usable on CPU-only builds for compile-tests of the
# header surface.
add_library(cstone_headers INTERFACE)
target_include_directories(cstone_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(cstone_headers INTERFACE cxx_std_20)
target_link_libraries(cstone_headers INTERFACE OpenMP::OpenMP_CXX)
add_library(ippl::cstone_headers ALIAS cstone_headers)

# GPU library: only when CUDA (or HIP) is the active language.
if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
  set(_CSTONE_GPU_SOURCES
    include/cstone/cuda/device_vector.cu
    include/cstone/focus/rebalance_gpu.cu
    include/cstone/focus/source_center_gpu.cu
    include/cstone/halos/gather_halos_gpu.cu
    include/cstone/primitives/primitives_gpu.cu
    include/cstone/sfc/sfc_gpu.cu
    include/cstone/traversal/collisions_gpu.cu
    include/cstone/traversal/groups_gpu.cu
    include/cstone/tree/csarray_gpu.cu
    include/cstone/tree/octree_gpu.cu)

  if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
    set_source_files_properties(${_CSTONE_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
  endif()

  add_library(cstone_gpu ${_CSTONE_GPU_SOURCES})
  target_include_directories(cstone_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
  target_compile_features(cstone_gpu PUBLIC cxx_std_20)
  target_link_libraries(cstone_gpu PUBLIC OpenMP::OpenMP_CXX)
  # OpenMP::OpenMP_CXX tags -fopenmp for CXX only. nvcc compiling .cu pulls
  # in cstone host headers that contain #pragma omp directives, so we also
  # need -fopenmp forwarded to the host compiler for CUDA TUs. This is a
  # COMPILE-only addition; we deliberately do NOT add -Xcompiler=-fopenmp
  # to link options, because Kokkos' launch-compiler wraps the device-link
  # through plain gcc which rejects nvcc-style flags.
  target_compile_options(cstone_gpu PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
  set_target_properties(cstone_gpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON)

  if(CSTONE_WITH_GPU_AWARE_MPI)
    target_compile_definitions(cstone_gpu INTERFACE CSTONE_HAVE_GPU_AWARE_MPI)
  endif()

  add_library(ippl::cstone_gpu ALIAS cstone_gpu)
endif()
