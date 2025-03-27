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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS NO)

cmake_minimum_required(VERSION 3.26)
project(exaDEM LANGUAGES C CXX)


# ===========================
# === CMake customization ===
# ===========================
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Coverage RelWithDebInfo Debug)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ==============================
# === Application version    ===
# ==============================
set(EXADEM_VERSION "exaDEM-1.1.0") 
message(STATUS "Project version : ${EXADEM_VERSION}")


# ======================================
# === exaDEM package specializations ===
# ======================================

# These values have to be set by exaNBody
set(EXADEM_MAX_VERTICES "9" CACHE STRING "Maximum number of vertices per polyhedron")
message(STATUS "Maximum number of vertices per polyhedron: ${EXADEM_MAX_VERTICES}, default is 9 vertices.")

set(XNB_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/include)
set(XNB_COMPILE_DEFINITIONS XNB_DOMAIN_SPECIFIC_FIELDS_INCLUDE=\<exaDEM/fields.h\> EXADEM_MAX_VERTICES=${EXADEM_MAX_VERTICES} EXADEM_VERSION=${EXADEM_VERSION})
set(XNB_APP_HAS_GRID_SPECIALIZATIONS ON)

# ======================================
# === ExaDEM Package Option          ===
# ======================================

option(USE_RSA "Add RSA Package" OFF)
if(USE_RSA)
	find_package(rsa_mpi REQUIRED)
	message( STATUS "rsa_mpi is used, header location: ${rsa_mpi_INCLUDE_DIRS}")
	message( STATUS "rsa_mpi is used, rsa_mpi_DIR: ${rsa_mpi_DIR}")
  list(APPEND XNB_INCLUDE_DIRECTORIES ${rsa_mpi_INCLUDE_DIRS})
endif()


# ======================================
# === exaNBody application framework ===
# ======================================
find_package(exaNBody)

# ======================================
# === add compiler option with CUDA  ===
# ======================================
if(ONIKA_BUILD_CUDA)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

# ==========================================
# === build application plugins and libs ===
# ==========================================
add_subdirectory(src)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/exaDEM DESTINATION include)

# =============================
# === check exaDEM examples ===
# =============================
add_subdirectory(example)

# =======================================
# === contributed tools and libraries ===
# =======================================
option(EXADEM_BUILD_CONTRIBS "Build contrib tools" ON)
if(EXADEM_BUILD_CONTRIBS)
  add_subdirectory(contribs)
endif()

# ===================================
# === Documentation               ===
# ===================================
option(EXADEM_BUILD_DOC "Build documentation" OFF)
if(EXADEM_BUILD_DOC)
  add_subdirectory(doc)
endif()


# ===========================================================
# === generate execution wrappers and environment scripts ===
# ===========================================================
GenerateEndUserExecEnv()

