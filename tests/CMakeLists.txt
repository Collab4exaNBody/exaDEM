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

# ==================================
# === exaDEM unit/regression tests ===
# ==================================
# Tests are sorted by plugin: tests/<plugin>/test_<name>.cpp
# Each file produces one executable registered as UnitTest_<plugin>_<name>.

enable_testing()

# plugin headers include each other (e.g. forcefield -> driver), all of them are visible
file(GLOB EXADEM_PLUGIN_INCLUDE_DIRS LIST_DIRECTORIES true ${CMAKE_SOURCE_DIR}/src/*/include)

function(exadem_add_unit_test PluginName TestName)
  set(TargetName exaDEM_test_${PluginName}_${TestName})
  add_executable(${TargetName} ${CMAKE_CURRENT_SOURCE_DIR}/test_${TestName}.cpp)
  target_include_directories(${TargetName} PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/common
    ${CMAKE_SOURCE_DIR}/src/${PluginName}/include
    ${EXADEM_PLUGIN_INCLUDE_DIRS}
    ${XNB_INCLUDE_DIRECTORIES})
  target_compile_definitions(${TargetName} PRIVATE ${XNB_COMPILE_DEFINITIONS})
  target_link_libraries(${TargetName} PRIVATE exanbCore)
  add_test(NAME UnitTest_${PluginName}_${TestName} COMMAND ${TargetName})
endfunction()

add_subdirectory(shape)
add_subdirectory(forcefield)
add_subdirectory(driver)
