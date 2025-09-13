# ================================
# === create exadem.cmake file ===
# ================================

set(XNB_APP_NAME ${CMAKE_PROJECT_NAME})
set(ONIKA_RUN_WRAPPER ${XNB_APP_NAME})
get_filename_component(XNB_ROOT_DIR ${CMAKE_INSTALL_PREFIX} ABSOLUTE)
set(XNB_CMAKE_PACKAGE ${CMAKE_BINARY_DIR}/exadem-config.cmake)
string(TIMESTAMP XNB_BUILD_DATE "%Y-%m-%d %Hh%M:%S")
file(WRITE ${XNB_CMAKE_PACKAGE} "# exaDEM CMake package (generated on ${XNB_BUILD_DATE})\n\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "# Main package configuration\n")
if(onika_DIR)
  file(APPEND ${XNB_CMAKE_PACKAGE} "set(onika_DIR ${onika_DIR})\n")
endif()
file(APPEND ${XNB_CMAKE_PACKAGE} "if(NOT onika_FOUND)\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "  find_package(onika REQUIRED)\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "endif()\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "set(XNB_ROOT_DIR ${XNB_ROOT_DIR})\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "set(XNB_APP_NAME \${CMAKE_PROJECT_NAME})\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "set(ONIKA_RUN_WRAPPER \${CMAKE_BINARY_DIR}/\${XNB_APP_NAME})\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "list(APPEND ONIKA_COMPILE_DEFINITIONS \${XNB_COMPILE_DEFINITIONS})\n")
file(APPEND ${XNB_CMAKE_PACKAGE} "list(APPEND ONIKA_INCLUDE_DIRS \${XNB_INCLUDE_DIRECTORIES})\n")
# configure optional DATA search directories
set(XNB_DATA_DIRS "${ONIKA_DEFAULT_DATA_DIRS}" CACHE STRING "Set of paths to search data files without relative paths")
file(APPEND ${XNB_CMAKE_PACKAGE} "set(XNB_DATA_DIRS \"${XNB_DATA_DIRS}\")\n")
