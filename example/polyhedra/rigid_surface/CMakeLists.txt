file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/octahedron.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("PolyhedraRigidSurfaceCompressiveMotion" "rigid_surface_compression_motion.msp" "regression_test_rigid_surface.msp")
