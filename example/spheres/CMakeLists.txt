enable_testing()

list(APPEND DLIST rotating-drum impose_velocity rigid-surface axial-stress quadratic-force-test mesh-stl movable-wall multimat)

foreach(DirExample IN LISTS DLIST)
  add_subdirectory(${DirExample})
endforeach()

if(USE_RSA)
  add_subdirectory(rsa)
endif()
