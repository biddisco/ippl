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
  add_executable(LandauDampingBH
    LandauDampingBH.cpp
    LandauDampingBHManager.cu)
  target_link_libraries(LandauDampingBH PRIVATE IPPL::ippl ippl::nbody)
  target_include_directories(LandauDampingBH PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(DisorderHeatingBH
    DisorderHeatingBH.cu
    DisorderHeatingBHManager.cu)
  target_link_libraries(DisorderHeatingBH PRIVATE IPPL::ippl ippl::nbody)
  target_include_directories(DisorderHeatingBH PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
