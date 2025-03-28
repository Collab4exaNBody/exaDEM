## add basic command tests 
add_test(NAME "Command_ExaDEM_Help" COMMAND ${CMAKE_BINARY_DIR}/exaDEM --help)
# add_test(NAME "Command_ExaDEM_Help_Default_Config" COMMAND ${CMAKE_BINARY_DIR}/exaDEM --help default-config)
# add_test(NAME "Command_ExaDEM_Help_Command_Line" COMMAND ${CMAKE_BINARY_DIR}/exaDEM --help command-line)
# add_test(NAME "Command_ExaDEM_Help_Plugins" COMMAND ${CMAKE_BINARY_DIR}/exaDEM --help plugins)
# add_test(NAME "Command_ExaDEM_Help_Show_Plugins" COMMAND ${CMAKE_BINARY_DIR}/exaDEM --help show-plugins)

