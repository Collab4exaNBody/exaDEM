# licensed to the apache software foundation (asf) under one
# or more contributor license agreements.  see the notice file
# distributed with this work for additional information
# regarding copyright ownership.  the asf licenses this file
# to you under the apache license, version 2.0 (the
# "license"); you may not use this file except in compliance
# with the license.  you may obtain a copy of the license at
# 
#   http://www.apache.org/licenses/license-2.0
# 
# unless required by applicable law or agreed to in writing,
# software distributed under the license is distributed on an
# "as is" basis, without warranties or conditions of any
# kind, either express or implied.  see the license for the
# specific language governing permissions and limitations
# under the license.
# ===============
# === Project ===
# ===============
cmake_minimum_required(VERSION 3.16)
project(exaDEM LANGUAGES C CXX)

# ===========================
# === CMake customization ===
# ===========================
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Coverage RelWithDebInfo Debug)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ==============================================
# === Set Some default values for polyhedra ====
# ==============================================
# These values have to be set by exaNBody
set(EXADEM_MAX_VERTICES "8" CACHE STRING "Maximum number of vertices per polyhedron")
message(STATUS "Maximum number of vertices per polyhedron: ${EXADEM_MAX_VERTICES}, default is 8 vertices.")
add_compile_options(-DEXADEM_MAX_VERTICES=${EXADEM_MAX_VERTICES})

# ======================================
# === ExaDEM Package Option          ===
# ======================================

option(USE_RSA "Add RSA Package" OFF)

if(USE_RSA)
	find_package(rsa_mpi REQUIRED)
	message( STATUS "rsa_mpi is used, header location: ${rsa_mpi_INCLUDE_DIRS}")
	message( STATUS "rsa_mpi is used, rsa_mpi_DIR: ${rsa_mpi_DIR}")
  include_directories(${rsa_mpi_INCLUDE_DIRS})
endif(USE_RSA)

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
set(EXASDEM_VERSION "exaDEM-1.1.0") 
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

# ======================================
# === add compiler option with CUDA  ===
# ======================================
if(XNB_BUILD_CUDA)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()


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

