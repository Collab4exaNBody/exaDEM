message(STATUS "Example polyhedron::triperiodic_compression")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/dodecahedron.shp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# sequential run only: check_compacity.py reads the analysis file written by this single run
set(TestName PolyhedraTriperiodicCompression)
add_test(NAME Example_ExaDEM_${TestName}
         COMMAND ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/triperiodic_compression.msp
                 ${CMAKE_CURRENT_SOURCE_DIR}/regression_test_triperiodic_compression.msp --omp_num_threads 1
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# the analysis file is appended to: it is removed before the run
add_test(NAME Example_ExaDEM_${TestName}_Clean
         COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/PolyhedraTriperiodicCompressionDir/ExaDEMAnalyses)
set_tests_properties(Example_ExaDEM_${TestName}_Clean PROPERTIES FIXTURES_SETUP ${TestName}_Clean)
set_tests_properties(Example_ExaDEM_${TestName} PROPERTIES FIXTURES_REQUIRED ${TestName}_Clean
                                                           FIXTURES_SETUP ${TestName})

# compacity reached at the end of the regression run (t = 2: 20000 steps, dt = 1e-4) is ~0.265
set(MinFinalCompacity 0.26)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME Example_ExaDEM_${TestName}_Check
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_compacity.py --dir ${CMAKE_CURRENT_BINARY_DIR}
                   --min-final-compacity ${MinFinalCompacity})
  set_tests_properties(Example_ExaDEM_${TestName}_Check PROPERTIES FIXTURES_REQUIRED ${TestName})
endif()
