enable_testing()

list(APPEND DLIST rotating-drum impose_velocity radial-stress rigid-surface axial-stress quadratic-force-test mesh-stl)

foreach(DirExample IN LISTS DLIST)
  add_subdirectory(${DirExample})
endforeach()
