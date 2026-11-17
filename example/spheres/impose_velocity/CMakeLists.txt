file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input_file_impose_velocity.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test("ImposeVelocity" "impose_velocity.msp")
add_omp_mpi_test_regression_file("ImposeVelocityHole" "impose_velocity_hole.msp" "regression_impose_velocity_hole.msp")
