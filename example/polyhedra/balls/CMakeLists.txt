file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/alpha3.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test("HexapodsBalls" "balls.msp")
