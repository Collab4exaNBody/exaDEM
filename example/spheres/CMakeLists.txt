enable_testing()

list(APPEND DLIST rotating-drum impose_velocity radial-stress rigid-surface axial-stress quadratic-force-test mesh-stl movable-wall)

foreach(DirExample IN LISTS DLIST)
  add_subdirectory(${DirExample})
endforeach()

if(USE_RSA)
  add_subdirectory(rsa)
endif()
