file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shapes.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test("InteractionOutput" "interaction_minimal.msp")
add_omp_mpi_test("WritePosVelOutput" "write_pos_vel_minimal.msp")
