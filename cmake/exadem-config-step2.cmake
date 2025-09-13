# ================================
# === create exadem.cmake file ===
# ================================

# optional search path configuration is per end user application
file(APPEND ${XNB_CMAKE_PACKAGE} "\n# Optional search directories for data files\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "set(\${XNB_APP_NAME}_DATA_DIRS \"${XNB_DATA_DIRS}\" CACHE STRING \"Set of paths to search data files without relative paths\")\n")

# install cmake package file to allow external project to find_package exaNBody
install(FILES ${XNB_CMAKE_PACKAGE} DESTINATION ${CMAKE_INSTALL_PREFIX})

