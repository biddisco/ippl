# -----------------------------------------------------------------------------
# alpine/CMakeLists.txt
#
# This module provides example physics applications using IPPL.
# -----------------------------------------------------------------------------

add_subdirectory(ExamplesWithoutPicManager)

function(add_alpine_example name)
  add_executable(${name} ${name}.cpp)

  target_link_libraries(${name} PRIVATE IPPL::ippl)

  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

if(IPPL_ENABLE_TESTS)
  # cmake-format: off

  set(LANDAU_RANKS 2)
  set(LANDAU_CSV "${PROJECT_BINARY_DIR}/alpine/data/FieldLandau_${LANDAU_RANKS}_manager.csv")

  # Generate a small cmake script so both init steps run as a single fixture test
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/LandauDampingInit.cmake"
    "file(MAKE_DIRECTORY \"${PROJECT_BINARY_DIR}/alpine/data\")\n"
    "file(REMOVE \"${LANDAU_CSV}\")\n"
  )
  add_test(NAME LandauDampingInit
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/LandauDampingInit.cmake")
  set_tests_properties(LandauDampingInit PROPERTIES FIXTURES_SETUP landau_fixture)

  # Add the test
  add_ippl_integration_test(LandauDamping 
    ARGS "16" "16" "16" "10000000" "25" "FFT" "0.01" "LeapFrog" "--overallocate" "2.0" "--info" "10"
    LABELS alpine integration
    NUM_PROCS ${LANDAU_RANKS}
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/alpine/")

  set_tests_properties(LandauDamping PROPERTIES FIXTURES_REQUIRED landau_fixture)
    
  # cmake-format: on
else()
  add_alpine_example(LandauDamping)
endif()

add_alpine_example(PenningTrap)
add_alpine_example(BumponTailInstability)

add_subdirectory(validation)
