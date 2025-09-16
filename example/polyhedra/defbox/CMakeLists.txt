file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/octahedron.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/xform_shear.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("PolyhedraDefBox" "shear_with_xform_dem.msp" "regression_test_shear_with_xform_dem.msp")
