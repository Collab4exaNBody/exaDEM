enable_testing()

#list(APPEND DLIST generator rotating_drum balls stl_mesh analyses)
list(APPEND DLIST generator balls stl_mesh analyses)
foreach(DirExample IN LISTS DLIST)
  add_subdirectory(${DirExample})
endforeach()
