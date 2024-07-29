file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/particles.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test("QuadraticForce" "QuadraticForceInput.msp")
