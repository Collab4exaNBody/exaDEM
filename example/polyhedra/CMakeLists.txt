enable_testing()

list(APPEND DLIST generator)

foreach(DirExample IN LISTS DLIST)
  add_subdirectory(${DirExample})
endforeach()
