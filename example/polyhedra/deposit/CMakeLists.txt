message(STATUS "Example polyhedron::deposit")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/mini_sample.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test("PolyhedraDepositAddXYZ" "add_xyz.msp")
