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
# Simulation-specific code lives next to its driver in test/BH/ — see
# test/BH/LandauInit.{hpp,cu}, test/BH/LandauDiagnostics.{hpp,cu},
# test/BH/DihDiagnostics.{hpp,cu}. The DIH focusing kernel is inlined into
# DisorderHeatingBHManager.hpp (its driver is .cu so nvcc sees the launch).
#
# The module's source files (.cu) follow design plan §G: native CUDA + cstone,
# no #include <Kokkos_Core.hpp> in the .cu/.cpp files. Kokkos appears only in
# the public headers as the type of consumer-facing unmanaged-View accessors.
# -----------------------------------------------------------------------------

add_library(ippl_nbody STATIC
  SphexaParticleContainer.cu
  SphexaBHSolver.cu
  LeapfrogStepper.cu
  PeriodicWrap.cu
  BeamDiagnostics.cu)

target_include_directories(ippl_nbody PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ippl>)

target_compile_features(ippl_nbody PUBLIC cxx_std_20)

target_link_libraries(ippl_nbody
  PUBLIC ippl::cstone_gpu
         ippl::ryoanji
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

install(FILES SphexaParticleContainer.hpp
              SphexaBHSolver.hpp
              LeapfrogStepper.hpp
              PeriodicWrap.hpp
              BeamDiagnostics.hpp
              NBodyManager.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ippl/NBody)
