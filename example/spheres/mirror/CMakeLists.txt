file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input_file_mirror.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("SphereMirrorBCS" "mirror.msp" "regression-test-mirror.msp")
