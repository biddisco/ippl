# -----------------------------------------------------------------------------
# src/NBody/CMakeLists.txt
#
# Producer code for the gridless Barnes-Hut module. Built only when
# IPPL_ENABLE_NBODY=ON (which already requires CUDA in IPPL_PLATFORMS — see
# top-level CMakeLists.txt).
#
# This library holds only *shared* infrastructure: container, BH solver,
# generic time-step primitives (leapfrog, periodic wrap), shared diagnostics
# (mean velocity, temperature), and the NBodyManager base class.
#
# Simulation-specific code lives next to its driver in test/BH/.
#
# Portability (mirrors sphexa's per-backend source selection):
#   - GPU build (CUDA or HIP): the device kernels live in .cu translation units,
#     retargeted to HIP when hipcc is the device compiler, linked against the
#     cstone/ryoanji GPU libraries.
#   - CPU build: the same modules are compiled from their .cpp counterparts
#     (inline OpenMP paths), linked against the header-only cstone/ryoanji
#     INTERFACE targets.
# Compute TUs are Kokkos-free; Kokkos appears only in the public container
# header as the type of the consumer-facing unmanaged-View accessor.
# -----------------------------------------------------------------------------

# Pure host orchestration (no device kernels): always compiled by the CXX
# compiler. On GPU builds USE_CUDA is propagated PUBLIC from cstone_gpu, selecting
# NBodyAcc=GpuTag and the GPU MultipoleHolder.
set(_NBODY_COMMON_SOURCES
  SphexaBHSolver.cpp)

# Device-kernel modules: .cu on GPU builds (retargeted to HIP under hipcc),
# inline-OpenMP .cpp on CPU builds.
if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
  set(_NBODY_KERNEL_SOURCES
    SphexaParticleContainer.cu
    LeapfrogStepper.cu
    PeriodicWrap.cu
    BeamDiagnostics.cu)
  if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
    set_source_files_properties(${_NBODY_KERNEL_SOURCES} PROPERTIES LANGUAGE HIP)
  endif()
  set(_NBODY_BACKEND_LIBS ippl::cstone_gpu ippl::ryoanji)
else()
  set(_NBODY_KERNEL_SOURCES
    SphexaParticleContainer.cpp
    LeapfrogStepper.cpp
    PeriodicWrap.cpp
    BeamDiagnostics.cpp)
  set(_NBODY_BACKEND_LIBS ippl::cstone_headers ippl::ryoanji_headers OpenMP::OpenMP_CXX)
endif()

add_library(ippl_nbody STATIC ${_NBODY_COMMON_SOURCES} ${_NBODY_KERNEL_SOURCES})

target_include_directories(ippl_nbody PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ippl>)

target_compile_features(ippl_nbody PUBLIC cxx_std_20)

target_link_libraries(ippl_nbody
  PUBLIC ${_NBODY_BACKEND_LIBS}
         Kokkos::kokkos)

# Note: CUDA_SEPARABLE_COMPILATION is intentionally NOT enabled on ippl_nbody.
# This TU includes <Kokkos_Core.hpp> for the View accessor type, and Kokkos'
# desul atomics emits a hard `#error Relocatable device code mode incompatible
# with desul atomics configuration` unless Kokkos was built with
# Kokkos_ENABLE_RELOCATABLE_DEVICE_CODE=ON. ippl_nbody has no device-side
# cross-TU calls that would need RDC, so we leave it off here.
set_target_properties(ippl_nbody PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_library(ippl::nbody ALIAS ippl_nbody)

install(FILES Accelerator.hpp
              BHPrecision.hpp
              SphexaParticleContainer.hpp
              GravityWrapper.hpp
              SphexaBHSolver.hpp
              LeapfrogStepper.hpp
              PeriodicWrap.hpp
              BeamDiagnostics.hpp
              NBodyManager.hpp
              ParticleAttribBase.hpp
              ParticleAttrib.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ippl/NBody)
