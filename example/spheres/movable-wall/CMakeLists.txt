file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input_file_movable_wall.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test("MoveWallSpheres" "movable_wall_mini.msp")
