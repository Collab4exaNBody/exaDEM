file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/aggregate.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cup.stl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("Aggregate" "aggregate.msp" "regression_test_aggregate.msp")
