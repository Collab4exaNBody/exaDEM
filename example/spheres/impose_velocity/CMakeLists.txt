file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input_file_impose_velocity.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test("ImposeVelocity" "impose_velocity.msp")
