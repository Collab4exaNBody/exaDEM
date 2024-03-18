
set(ExampleName "ImposeVelocity")
set(FileName "impose_velocity.msp")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input_file_impose_velocity.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_test(Example_ExaDEM_${ExampleName} ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/${FileName} --omp_num_threads 1)
add_test(Example_ExaDEM_${ExampleName}_omp ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/${FileName} --omp_num_threads 4)
add_test(Example_ExaDEM_${ExampleName}_mpi mpirun -n 4 ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/${FileName} --omp_num_threads 1)
