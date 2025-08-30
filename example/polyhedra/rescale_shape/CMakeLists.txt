file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shapes.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("ScalePolyhdraFCC" "scale_2_shapes_fcc.msp" "regression_test_common.msp")

if(USE_RSA)
	add_omp_mpi_test_regression_file("ScalePolyhdraRSA" "scale_2_shapes_rsa.msp" "regression_test_common.msp")
endif(USE_RSA)
