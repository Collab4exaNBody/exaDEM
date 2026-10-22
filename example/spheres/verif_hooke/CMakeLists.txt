file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test1_two_spheres.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test2_sphere_plane.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test3_restitution_sweep.xyz DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

find_package(Python3 COMPONENTS Interpreter QUIET)

# Runs exaDEM on <msp> (serial), then checks the output against the exact
# Hooke solution with check_hooke.py <key>.
function(add_verif_hooke_test TestName FileName Key)
  add_test(NAME Example_ExaDEM_${TestName}
           COMMAND ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/${FileName} --omp_num_threads 1
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(Example_ExaDEM_${TestName} PROPERTIES FIXTURES_SETUP ${TestName})
  if(Python3_Interpreter_FOUND)
    add_test(NAME Example_ExaDEM_${TestName}_Check
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_hooke.py ${Key}
                     --dir ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(Example_ExaDEM_${TestName}_Check PROPERTIES FIXTURES_REQUIRED ${TestName})
  endif()
endfunction()

add_verif_hooke_test(VerifHookeTest1TwoSpheres "test1_two_spheres.msp" test1)
add_verif_hooke_test(VerifHookeTest2SpherePlane "test2_sphere_plane.msp" test2)
add_verif_hooke_test(VerifHookeTest3RestitutionSweep "test3_restitution_sweep.msp" test3)

# damprate values of test3 vs scripts/tools/compute_damp_rate.py (no exaDEM run)
if(Python3_Interpreter_FOUND)
  add_test(NAME Example_ExaDEM_VerifHookeDampRate
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/verify_damp_rate.py)
endif()
