file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cylinder.stl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("SphereCylinderStlTestFile" "cylinder_stl.msp" "regression_test_cylinder_stl.msp")
