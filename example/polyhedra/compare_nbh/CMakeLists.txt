# Global tests of the polyhedron neighbor search: each one runs nbh_polyhedron (reference) and another
# neighbor search operator from the same input every time the neighbor lists are rebuilt, and stops with an
# error if the classifiers differ (see data/config/config_compare_nbh_gpu.msp).
add_omp_mpi_test_regression_file("PolyhedraCompareNbhOBBTree" "rice_bowl_scene.msp" "compare_obb_tree.msp")

# nbh_polyhedron_gpu needs a CUDA build.
option(EXADEM_ENABLE_GPU_COMPARISON_TESTS "Compare nbh_polyhedron_gpu with nbh_polyhedron in ctest" ${ONIKA_BUILD_CUDA})
if(EXADEM_ENABLE_GPU_COMPARISON_TESTS)
  add_omp_mpi_test_regression_file("PolyhedraCompareNbhGPU" "rice_bowl_scene.msp" "compare_gpu.msp")
endif()
