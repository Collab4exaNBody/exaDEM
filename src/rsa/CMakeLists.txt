#   Licensed to the Apache Software Foundation (ASF) under one
#   or more contributor license agreements.  See the NOTICE file
#   distributed with this work for additional information
#   regarding copyright ownership.  The ASF licenses this file
#   to you under the Apache License, Version 2.0 (the
#   "License"); you may not use this file except in compliance
#   with the License.  You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing,
#   software distributed under the License is distributed on an
#   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied.  See the License for the
#   specific language governing permissions and limitations
#   under the License.

find_package(rsa_mpi REQUIRED)
message( STATUS "rsa_mpi is used, header location: ${rsa_mpi_INCLUDE_DIRS}")
message( STATUS "rsa_mpi is used, rsa_mpi_DIR: ${rsa_mpi_DIR}")
include_directories(${rsa_mpi_INCLUDE_DIRS})

set(exadem_rsa_LINK_LIBRARIES exanbIO exanbDefBox exanbParticleNeighbors exadem_numerical_scheme exadem_shape)
xnb_add_plugin(exadem_rsa ${CMAKE_CURRENT_SOURCE_DIR})
