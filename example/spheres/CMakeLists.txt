enable_testing()

list(APPEND DLIST rotating-drum ball impose_velocity rigid-surface axial-stress quadratic-force-test mesh-stl movable-wall multimat mixer mirror cylinder_stl piston)

foreach(DirExample IN LISTS DLIST)
  add_subdirectory(${DirExample})
endforeach()

if(USE_RSA)
  add_subdirectory(rsa)
endif()
