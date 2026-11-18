message(STATUS "Example polyhedron::triperiodic_compression")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/dodecahedron.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(TestName PolyhedraTriperiodicCompression)
add_omp_mpi_test_regression_file(${TestName} "triperiodic_compression.msp" "regression_test_triperiodic_compression.msp")

# the three runs append to the same analysis file: it is removed first and the runs are serialized
add_test(NAME Example_ExaDEM_${TestName}_Clean
         COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/PolyhedraTriperiodicCompressionDir/ExaDEMAnalyses)
set_tests_properties(Example_ExaDEM_${TestName}_Clean PROPERTIES FIXTURES_SETUP ${TestName}_Clean)
set_tests_properties(Example_ExaDEM_${TestName} Example_ExaDEM_${TestName}_omp Example_ExaDEM_${TestName}_mpi
                     PROPERTIES FIXTURES_REQUIRED ${TestName}_Clean FIXTURES_SETUP ${TestName}
                                RESOURCE_LOCK ${TestName})

# compacity reached at the end of the regression run (t = 2: 20000 steps, dt = 1e-4) is ~0.265
set(MinFinalCompacity 0.26)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME Example_ExaDEM_${TestName}_Check
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_compacity.py --dir ${CMAKE_CURRENT_BINARY_DIR}
                   --min-final-compacity ${MinFinalCompacity})
  set_tests_properties(Example_ExaDEM_${TestName}_Check PROPERTIES FIXTURES_REQUIRED ${TestName})
endif()
