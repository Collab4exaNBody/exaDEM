enable_testing()


list(APPEND DLIST generator rotating_drum balls stl_mesh analyses multimat)

foreach(DirExample IN LISTS DLIST)
  add_subdirectory(${DirExample})
endforeach()
