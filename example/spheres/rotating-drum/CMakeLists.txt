add_test(Example_ExaDEM_Rotating_Drum ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/rotating-drum.msp --omp_num_threads 1)
add_test(Example_ExaDEM_Rotating_Drum_omp ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/rotating-drum.msp --omp_num_threads 4)
add_test(Example_ExaDEM_Rotating_Drum_mpi mpirun -n 4 ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/rotating-drum.msp --omp_num_threads 1)
