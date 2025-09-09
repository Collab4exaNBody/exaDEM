file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shapes.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cylinder.stl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("STLCylinderMultiShape" "stl_cylinder_multi_shp.msp" "regression_test_stl_cylinder_multi_shp.msp")
