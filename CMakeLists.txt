# ===============
# === Project ===
# ===============
cmake_minimum_required(VERSION 3.16)
project(exaDEM LANGUAGES C CXX)

# ===========================
# === CMake customization ===
# ===========================
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release RelWithDebInfo Debug)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ==============================================
# === Set Some default values for polyhedra ====
# ==============================================
# These values have to be set by exaNBody
set(EXADEM_MAX_VERTICES "8" CACHE STRING "Maximum number of vertices per polyhedron")
message(STATUS "Maximum number of vertices per polyhedron: ${EXADEM_MAX_VERTICES}")
add_compile_options(-DEXADEM_MAX_VERTICES=${EXADEM_MAX_VERTICES})
#set(USTAMP_COMPILE_DEFINITIONS ${USTAMP_COMPILE_DEFINITIONS} "EXADEM_MAX_VERTICES=${EXADEM_MAX_VERTICES}")
message(STATUS ${USTAMP_COMPILE_DEFINITIONS})

# ======================================
# === exaNBody application framework ===
# ======================================
find_package(exaNBody)

# =================================
# === Host System configuration ===
# =================================
# find command that gives OS description string
find_file(CCC_OS_CMD NAMES ccc_os cea_os uname PATHS /ccc/local/products/sr/bin /usr/local/sr/bin /ccc/products/ccc_users_env/bin ENV PATH)
if(CCC_OS_CMD)
  execute_process(COMMAND ${CCC_OS_CMD} OUTPUT_VARIABLE HOST_OS OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# =========================
# === Custom submodules ===
# =========================
foreach(esm ${EXASTAMP_SUBMODULES})
    if(EXISTS ${CMAKE_SOURCE_DIR}/${esm}/config.cmake)
        message(STATUS "load sub module configuration ${CMAKE_SOURCE_DIR}/${esm}/config.cmake")
        include(${CMAKE_SOURCE_DIR}/${esm}/config.cmake)
    endif()
endforeach()

# ==============================
# === Application version    ===
# ==============================
set(EXASDEM_VERSION "exaDEM-3.3") 
message(STATUS "Project version : ${EXASDEM_VERSION}")

# ==================================
# === environment setup commands ===
# ==================================
set(PROJECT_SETUP_ENV_COMMANDS "" CACHE STRING "Environment setup commands") 

# ====================================
# === CCC supercomputing facility  ===
# ====================================
option(EXASTAMP_CCC_USER "Use CCC job resource interface" OFF)
if(EXASTAMP_CCC_USER)
  set(CCC_USER_DEFINITIONS "-D__use_lib_ccc_user=1" CACHE STRING "ccc_user macro definitions") 
  set(CCC_USER_INCLUDE_DIRS "" CACHE STRING "ccc_user include dir") 
  set(CCC_USER_LIBRARIES "ccc_user" CACHE STRING "ccc_user lib") 
  set(XNB_APP_DEFINITIONS ${XNB_APP_DEFINITIONS} ${CCC_USER_DEFINITIONS})
  set(XNB_APP_INCLUDE_DIRS ${XNB_APP_INCLUDE_DIRS} ${CCC_USER_INCLUDE_DIRS})
  set(XNB_APP_LIBRARIES ${XNB_APP_LIBRARIES} ${CCC_USER_LIBRARIES})
endif()

# ==============================
# === test data directory    ===
# ==============================
include(DataDirectoryMD5)
set(EXASTAMP_TEST_DATA_DIR "./data" CACHE STRING "Path where to find input data files used by regression tests")
set(EXASTAMP_TEST_DATA_MD5_FILE ${CMAKE_SOURCE_DIR}/data/data_dir.md5)
set(EXASTAMP_TEST_DATA_DIR_STATUS ${CMAKE_CURRENT_BINARY_DIR}/data_dir_checked)
option(EXASTAMP_CHECK_DATA_DIR_MD5 "Check test data dir integrity" OFF)
if(EXISTS ${EXASTAMP_TEST_DATA_DIR_STATUS})
  file(READ ${EXASTAMP_TEST_DATA_DIR_STATUS} EXASTAMP_TEST_DATA_DIR_CHECKED)
else()
  set(EXASTAMP_TEST_DATA_DIR_CHECKED OFF)
endif()

if(IS_DIRECTORY ${EXASTAMP_TEST_DATA_DIR} AND EXASTAMP_CHECK_DATA_DIR_MD5 AND NOT EXASTAMP_TEST_DATA_DIR_CHECKED)
  CheckDirectoryMD5(${EXASTAMP_TEST_DATA_DIR} ${EXASTAMP_TEST_DATA_MD5_FILE} EXASTAMP_TEST_DATA_DIR_CHECKED)
  file(WRITE ${EXASTAMP_TEST_DATA_DIR_STATUS} ${EXASTAMP_TEST_DATA_DIR_CHECKED})
endif()


# =======================================
# === build exaNBody plugins and main ===
# =======================================
exaNBodyStartApplication()


# ==========================================
# === build application plugins and libs ===
# ==========================================
add_subdirectory(src)

# =============================
# === check exaDEM examples ===
# =============================
add_subdirectory(example)

# ==================================================
# === exaNBody plugins and unit tests final pass ===
# ==================================================
exaNBodyFinalizeApplication()

# =======================================
# === contributed tools and libraries ===
# =======================================
option(XNB_BUILD_CONTRIBS "Build contrib tools" ON)
if(XNB_BUILD_CONTRIBS)
  add_subdirectory(contribs)
endif()

# ===================================
# === Documentation               ===
# ===================================
option(EXASTAMP_BUILD_DOC "Build documentation" OFF)
if(EXASTAMP_BUILD_DOC)
  add_subdirectory(doc)
endif()

