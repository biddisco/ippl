# -----------------------------------------------------------------------------
# test/BH/CMakeLists.txt
#
# Integration tests for BH functionality. Each executable bundles its own
# simulation-specific .cu sources directly — only shared infrastructure (the
# container, BH solver, leapfrog, periodic wrap, shared diagnostics) lives in
# the ippl::nbody library.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding index test found in ${_relPath}")

if(IPPL_ENABLE_NBODY)
  # Pure-host drivers (.cpp) on both backends; the simulation-specific kernels
  # live in the per-backend manager TU: .cu on GPU (curand IC, thrust reductions,
  # focusing kernel), inline-OpenMP .cpp on CPU.
  if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    set(_LANDAU_MGR LandauDampingBHManager.cu)
    set(_DIH_MGR    DisorderHeatingBHManager.cu)
    if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
      set_source_files_properties(${_LANDAU_MGR} ${_DIH_MGR} PROPERTIES LANGUAGE HIP)
    endif()
  else()
    set(_LANDAU_MGR LandauDampingBHManager.cpp)
    set(_DIH_MGR    DisorderHeatingBHManager.cpp)
  endif()

  add_executable(LandauDampingBH
    LandauDampingBH.cpp
    ${_LANDAU_MGR})
  target_link_libraries(LandauDampingBH PRIVATE IPPL::ippl ippl::nbody)
  target_include_directories(LandauDampingBH PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(DisorderHeatingBH
    DisorderHeatingBH.cpp
    ${_DIH_MGR})
  target_link_libraries(DisorderHeatingBH PRIVATE IPPL::ippl ippl::nbody)
  target_include_directories(DisorderHeatingBH PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
