message(STATUS "Example polyhedron::f3p")

# Three-point bending (flexion 3 points) of a beam made of 1000 fragments.
# Not registered as a test: the simulation is too long.
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/beam_fragments.shp ${CMAKE_CURRENT_SOURCE_DIR}/beam_fragments.txt ${CMAKE_CURRENT_SOURCE_DIR}/support_left.shp ${CMAKE_CURRENT_SOURCE_DIR}/support_right.shp ${CMAKE_CURRENT_SOURCE_DIR}/punch.shp ${CMAKE_CURRENT_SOURCE_DIR}/f3p.msp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# weak_scaling/: generator of wide f3p beams (N copies of a periodic sample along y) for weak scaling
# studies; run make_f3p_weak_scaling.py from the source tree (see weak_scaling/README.md).
