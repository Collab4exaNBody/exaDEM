enable_testing()

function(add_omp_mpi_test ExampleName FileName)
	add_test(Example_ExaDEM_${ExampleName} ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/${FileName} --omp_num_threads 1)
	add_test(Example_ExaDEM_${ExampleName}_omp ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/${FileName} --omp_num_threads 2)
	add_test(Example_ExaDEM_${ExampleName}_mpi mpirun -n 2 ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/${FileName} --omp_num_threads 1)
endfunction(add_omp_mpi_test)


add_subdirectory(commands)
add_subdirectory(spheres)
add_subdirectory(polyhedra)
