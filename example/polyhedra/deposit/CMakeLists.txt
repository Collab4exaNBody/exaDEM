message(STATUS "Example polyhedron::deposit")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/mini_sample.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("PolyhedraDepositAddXYZ" "add_xyz.msp" "regression_add_xyz.msp")