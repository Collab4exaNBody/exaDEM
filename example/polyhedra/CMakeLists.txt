enable_testing()

list(APPEND DLIST generator rotating_drum)

foreach(DirExample IN LISTS DLIST)
  add_subdirectory(${DirExample})
endforeach()
