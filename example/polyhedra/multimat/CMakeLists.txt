file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/alpha3.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/octahedron.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test("OctahedraRotatingDrumMultiMat" "rotating-multimat_mini.msp")
