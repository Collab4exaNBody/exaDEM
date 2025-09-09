file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/piston_haut.stl ${CMAKE_CURRENT_SOURCE_DIR}/piston_bas.stl ${CMAKE_CURRENT_SOURCE_DIR}/regression_test_restart_file.dump DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if(USE_RSA)
	add_omp_mpi_test_regression_file("SphereInitPiston" "init_piston.msp" "regression_test_init_piston.msp")
endif(USE_RSA)
add_omp_mpi_test_regression_file("SphereLinearForcePiston" "piston_linear_force_motion.msp" "regression_test_common_parameters.msp")
add_omp_mpi_test_regression_file("SphereLinearCompressionPiston" "piston_linear_compression_motion.msp" "regression_test_common_parameters.msp")
