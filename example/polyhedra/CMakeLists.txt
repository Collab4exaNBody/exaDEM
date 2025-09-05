enable_testing()


list(APPEND DLIST generator rotating_drum balls stl_mesh analyses multimat conf_rockable)

foreach(DirExample IN LISTS DLIST)
  add_subdirectory(${DirExample})
endforeach()
