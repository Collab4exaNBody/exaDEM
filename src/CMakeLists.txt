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

# ===================================
# === exaDEM application package  ===
# ===================================

add_subdirectory(grid)
add_subdirectory(shape)
add_subdirectory(field)
add_subdirectory(numericalscheme)
add_subdirectory(driver)
add_subdirectory(forcefield)
add_subdirectory(interaction)
add_subdirectory(analysis)
add_subdirectory(io)
add_subdirectory(sphere)
add_subdirectory(polyhedron)


if(USE_RSA)
	add_subdirectory(rsa)
endif()

