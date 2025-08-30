enable_testing()

function(add_omp_mpi_test ExampleName FileName)
  set(TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${FileName}")
	add_test(Example_ExaDEM_${ExampleName} ${CMAKE_BINARY_DIR}/exaDEM ${TEST_FILES} --omp_num_threads 1)
	add_test(Example_ExaDEM_${ExampleName}_omp ${CMAKE_BINARY_DIR}/exaDEM ${TEST_FILES} --omp_num_threads 2)
	add_test(Example_ExaDEM_${ExampleName}_mpi mpirun -n 2 ${CMAKE_BINARY_DIR}/exaDEM ${TEST_FILES} --omp_num_threads 1)
endfunction(add_omp_mpi_test)

function(add_omp_mpi_test_regression_file ExampleName FileName RegressionFile)
  set(TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${FileName}")
  list(APPEND TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${RegressionFile}")
	add_test(Example_ExaDEM_${ExampleName} ${CMAKE_BINARY_DIR}/exaDEM ${TEST_FILES} --omp_num_threads 1)
	add_test(Example_ExaDEM_${ExampleName}_omp ${CMAKE_BINARY_DIR}/exaDEM ${TEST_FILES} --omp_num_threads 2)
	add_test(Example_ExaDEM_${ExampleName}_mpi mpirun -n 2 ${CMAKE_BINARY_DIR}/exaDEM ${TEST_FILES} --omp_num_threads 1)
endfunction(add_omp_mpi_test_regression_file)


add_subdirectory(commands)
add_subdirectory(spheres)
add_subdirectory(polyhedra)
