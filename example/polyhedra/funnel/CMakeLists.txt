file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/alpha3.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/box.stl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/funnel.stl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("FunnelSimulaion" "funnel.msp" "regression_test_funnel.msp")
