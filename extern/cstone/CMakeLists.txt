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

# cstone's domain/halo headers include <mpi.h>; MPI is a hard dependency of
# any consumer that includes those headers (i.e. anything beyond pure-CPU
# header-only utilities). Propagate as PUBLIC/INTERFACE.
find_package(MPI REQUIRED COMPONENTS CXX)

# Header-only target: usable on CPU-only builds for compile-tests of the
# header surface.
add_library(cstone_headers INTERFACE)
target_include_directories(cstone_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(cstone_headers INTERFACE cxx_std_20)
target_link_libraries(cstone_headers INTERFACE OpenMP::OpenMP_CXX MPI::MPI_CXX)
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
  target_link_libraries(cstone_gpu PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX)
  # cstone host headers (assignment.hpp, halos.hpp) guard their GPU-impl
  # #include lines behind `#ifdef USE_CUDA`. Upstream sets this PRIVATE on
  # every CUDA-build target (e.g. main/src/sphexa/CMakeLists.txt:34); we
  # propagate it PUBLIC from cstone_gpu so any consumer that links cstone_gpu
  # automatically sees the GPU function declarations.
  target_compile_definitions(cstone_gpu PUBLIC USE_CUDA)
  # OpenMP::OpenMP_CXX tags -fopenmp for CXX only. nvcc compiling .cu pulls
  # in cstone host headers that contain #pragma omp directives, so we also
  # need -fopenmp forwarded to the host compiler for CUDA TUs. This is a
  # COMPILE-only addition; we deliberately do NOT add -Xcompiler=-fopenmp
  # to link options, because Kokkos' launch-compiler wraps the device-link
  # through plain gcc which rejects nvcc-style flags.
  #
  # --extended-lambda enables nvcc support for __host__ __device__ lambdas,
  # which both cstone and IPPL/Kokkos use extensively. Upstream cstone sets
  # this globally via CMAKE_CUDA_FLAGS; we propagate it PUBLIC from cstone_gpu
  # so any consumer-side CUDA TU inherits it (e.g. our unit tests that include
  # IPPL headers with __host__ __device__ Kokkos lambdas).
  target_compile_options(cstone_gpu PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  # CUDA_SEPARABLE_COMPILATION (-rdc=true) is deliberately OFF: matches
  # sphexa's upstream cstone build and lets nvcc whole-program-inline the
  # warp-cooperative primitives (shflSync, warpMin/Max, etc.) that
  # ryoanji::traversal_gpu.cu pulls in from cstone headers.
  set_target_properties(cstone_gpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

  if(CSTONE_WITH_GPU_AWARE_MPI)
    target_compile_definitions(cstone_gpu INTERFACE CSTONE_HAVE_GPU_AWARE_MPI)
  else()
    message(WARNING
      "cstone_gpu is being built without CUDA-aware MPI "
      "(CSTONE_WITH_GPU_AWARE_MPI=OFF). Every halo exchange, focus-tree "
      "exchange and domain sync will stage device buffers through host "
      "memory via cudaMemcpy D2H/H2D around each MPI call. "
      "Pass -DCSTONE_WITH_GPU_AWARE_MPI=ON when the MPI implementation "
      "supports device pointers (e.g. Cray MPICH with "
      "MPICH_GPU_SUPPORT_ENABLED=1, OpenMPI built with CUDA, MVAPICH2-GDR).")
  endif()

  add_library(ippl::cstone_gpu ALIAS cstone_gpu)
endif()
