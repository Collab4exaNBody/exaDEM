
set(ExampleName "ShearWithXform")
set(FileName "shear_with_xform_dem.msp")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/735.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/xform_test.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/xform_shear.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_test(Example_ExaDEM_${ExampleName} ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/${FileName} --omp_num_threads 1)
add_test(Example_ExaDEM_${ExampleName}_omp ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/${FileName} --omp_num_threads 2)
add_test(Example_ExaDEM_${ExampleName}_mpi mpirun -n 2 ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/${FileName} --omp_num_threads 1)
