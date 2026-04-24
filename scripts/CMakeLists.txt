enable_testing()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(PYTHON_SCRIPTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# -----------------------------
# Helper function
# -----------------------------
function(add_pytest NAME FILE)
    add_test(
        NAME ${NAME}
        COMMAND ${Python3_EXECUTABLE} -m pytest -v ${FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    set_tests_properties(${NAME} PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PYTHON_SCRIPTS_DIR}"
    )
endfunction()

# -----------------------------
# Individual tests
# -----------------------------
add_pytest(test_compute_clusters_stresses_tensors
    test/test_compute_clusters_stresses_tensors.py
)

add_pytest(test_io_rockable
    test/test_io_rockable.py
)

add_pytest(test_tess2rockable
    test/test_tess2rockable.py
)

# -----------------------------
# Global test (CI friendly)
# -----------------------------
add_test(
    NAME pytest_all
    COMMAND ${Python3_EXECUTABLE} -m pytest -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(pytest_all PROPERTIES
    ENVIRONMENT "PYTHONPATH=${PYTHON_SCRIPTS_DIR}"
)
