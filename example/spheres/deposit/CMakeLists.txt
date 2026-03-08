message(STATUS "Example spheres::deposit")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input_deposit_xyz.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("SpheresDepositAddXYZ" "add_xyz.msp" "regression_add_xyz.msp")
