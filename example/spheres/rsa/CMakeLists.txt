add_test(Example_ExaDEM_RSA ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/rsa.msp --omp_num_threads 1)
add_test(Example_ExaDEM_RSA_omp ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/rsa.msp --omp_num_threads 2)
add_test(Example_ExaDEM_RSA_mpi mpirun -n 2 ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/rsa.msp --omp_num_threads 1)
