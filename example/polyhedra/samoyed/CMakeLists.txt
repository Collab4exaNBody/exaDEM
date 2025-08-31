file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/samoyed.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cup.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("Samoyed" "samoyed.msp" "regression_test_samoyed.msp")
