add_test(NAME Example_ExaDEM_ToolsConvExaDEMToRockable
         COMMAND ${CMAKE_SOURCE_DIR}/scripts/tools/ConvExaDEMToRockable
                 ${CMAKE_CURRENT_SOURCE_DIR}/data/CheckpointFiles/file.dump
                 ${CMAKE_CURRENT_BINARY_DIR}/test_conf.conf
                 ${CMAKE_CURRENT_SOURCE_DIR}/data/CheckpointFiles/RestartShapeFile.shp)
set_tests_properties(Example_ExaDEM_ToolsConvExaDEMToRockable PROPERTIES
  ENVIRONMENT "EXADEM_BIN=${CMAKE_BINARY_DIR}/exaDEM")

add_test(NAME Example_ExaDEM_ToolsConvExaDEMToXYZ
         COMMAND ${CMAKE_SOURCE_DIR}/scripts/tools/ConvExaDEMToXYZ
                 ${CMAKE_CURRENT_SOURCE_DIR}/data/CheckpointFiles/file.dump
                 ${CMAKE_CURRENT_BINARY_DIR}/test_out.xyz
                 ${CMAKE_CURRENT_SOURCE_DIR}/data/CheckpointFiles/RestartShapeFile.shp)
set_tests_properties(Example_ExaDEM_ToolsConvExaDEMToXYZ PROPERTIES
  ENVIRONMENT "EXADEM_BIN=${CMAKE_BINARY_DIR}/exaDEM")

set(RESTART_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/restart)
file(MAKE_DIRECTORY ${RESTART_TEST_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/restart/shape.shp DESTINATION ${RESTART_TEST_DIR})

add_test(NAME Example_ExaDEM_ToolsRestartGenerate
         COMMAND ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/restart/generate.msp
         WORKING_DIRECTORY ${RESTART_TEST_DIR})

add_test(NAME Example_ExaDEM_ToolsRestart
         COMMAND ${CMAKE_BINARY_DIR}/exaDEM ${CMAKE_CURRENT_SOURCE_DIR}/restart/restart.msp
         WORKING_DIRECTORY ${RESTART_TEST_DIR})
set_tests_properties(Example_ExaDEM_ToolsRestart PROPERTIES
  DEPENDS Example_ExaDEM_ToolsRestartGenerate)
