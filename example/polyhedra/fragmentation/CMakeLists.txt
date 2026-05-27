message(STATUS "Example polyhedron::fragmentation")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/two_fragments.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/two_fragments.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input_data_beam_two_particles.txt ${CMAKE_CURRENT_SOURCE_DIR}/planUp.shp ${CMAKE_CURRENT_SOURCE_DIR}/planDo.shp ${CMAKE_CURRENT_SOURCE_DIR}/beam.shp	DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_omp_mpi_test_regression_file("PolyhedraFragmentationTwoFragmentsBreak" "two_fragments_break.msp" "regression_test_two_fragments.msp")
add_omp_mpi_test_regression_file("PolyhedraFragmentationTwoFragmentsUnbreak" "two_fragments_unbreak.msp" "regression_test_two_fragments.msp")
add_omp_mpi_test("PolyhedraFragmentationBeamBreak" "beam_two_particles_break.msp")
