file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input_file DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test("ReadConfRockable" "read_conf_518_poly.msp")
