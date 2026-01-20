enable_testing()

list(APPEND DLIST generator rotating_drum balls stl_mesh analyses multimat conf_rockable rescale_shape samoyed aggregate defbox stl_cylinder funnel rigid_surface imposed_driver_moment fragmentation)

foreach(DirExample IN LISTS DLIST)
  add_subdirectory(${DirExample})
endforeach()
