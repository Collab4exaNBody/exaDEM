# Non-regression test for scripts/tools/ConvExaDEMToRockable, run against the sample checkpoint
# files in data/CheckpointFiles/ (a real .dump + its RestartShapeFile.shp).
add_test(NAME Example_ExaDEM_ToolsConvExaDEMToRockable
         COMMAND ${CMAKE_SOURCE_DIR}/scripts/tools/ConvExaDEMToRockable
                 ${CMAKE_CURRENT_SOURCE_DIR}/data/CheckpointFiles/file.dump
                 ${CMAKE_CURRENT_BINARY_DIR}/test_conf.conf
                 ${CMAKE_CURRENT_SOURCE_DIR}/data/CheckpointFiles/RestartShapeFile.shp)
set_tests_properties(Example_ExaDEM_ToolsConvExaDEMToRockable PROPERTIES
  ENVIRONMENT "EXADEM_BIN=${CMAKE_BINARY_DIR}/exaDEM")

# Non-regression test for scripts/tools/ConvExaDEMToXYZ, same sample checkpoint files.
add_test(NAME Example_ExaDEM_ToolsConvExaDEMToXYZ
         COMMAND ${CMAKE_SOURCE_DIR}/scripts/tools/ConvExaDEMToXYZ
                 ${CMAKE_CURRENT_SOURCE_DIR}/data/CheckpointFiles/file.dump
                 ${CMAKE_CURRENT_BINARY_DIR}/test_out.xyz
                 ${CMAKE_CURRENT_SOURCE_DIR}/data/CheckpointFiles/RestartShapeFile.shp)
set_tests_properties(Example_ExaDEM_ToolsConvExaDEMToXYZ PROPERTIES
  ENVIRONMENT "EXADEM_BIN=${CMAKE_BINARY_DIR}/exaDEM")
