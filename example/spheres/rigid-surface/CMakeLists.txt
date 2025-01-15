file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input_file_rigid_surface.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test("RigidSurface" "rigid_surface_minimal.msp")
