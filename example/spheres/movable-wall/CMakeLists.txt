file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input_file_movable_wall.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("MoveWallSpheres" "movable_wall.msp" "regression_test_movable_wall.msp")
