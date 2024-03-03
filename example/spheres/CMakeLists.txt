enable_testing()

list(APPEND DLIST rotating-drum)
#list(APPEND DLIST rotating-drum rigid-surface radial-stress)

foreach(DirExample IN LISTS DLIST)
  add_subdirectory(${DirExample})
endforeach()
