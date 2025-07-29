file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/external_mixer_from_chrono_dem.stl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/internal_mixer_from_chrono_dem.stl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/mixer_common.msp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test("MixerWithSpheres" "mixer_minimal.msp")
