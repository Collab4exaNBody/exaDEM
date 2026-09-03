file(WRITE ${CMAKE_BINARY_DIR}/dump_inspector
  "#!/bin/bash\n"
  "# dump_inspector: prints a .dump checkpoint file's header. See scripts/tools/dump_inspector.sh\n"
  "export EXADEM_BIN=\"${CMAKE_BINARY_DIR}/exaDEM\"\n"
  "export DUMP_INSPECTOR_MSP=\"${CMAKE_CURRENT_SOURCE_DIR}/dump_inspector.msp\"\n"
  "exec \"${CMAKE_CURRENT_SOURCE_DIR}/dump_inspector.sh\" \"$@\"\n")
file(CHMOD ${CMAKE_BINARY_DIR}/dump_inspector
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# ConvExaDEMToRockable: same pattern, for scripts/tools/ConvExaDEMToRockable.
file(WRITE ${CMAKE_BINARY_DIR}/ConvExaDEMToRockable
  "#!/bin/bash\n"
  "# ConvExaDEMToRockable: exports a .dump checkpoint file's particles to a Rockable .conf. See "
  "scripts/tools/ConvExaDEMToRockable\n"
  "export EXADEM_BIN=\"${CMAKE_BINARY_DIR}/exaDEM\"\n"
  "export DUMP_TO_ROCKABLE_MSP=\"${CMAKE_CURRENT_SOURCE_DIR}/dump_to_rockable.msp\"\n"
  "exec \"${CMAKE_CURRENT_SOURCE_DIR}/ConvExaDEMToRockable\" \"$@\"\n")
file(CHMOD ${CMAKE_BINARY_DIR}/ConvExaDEMToRockable
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
