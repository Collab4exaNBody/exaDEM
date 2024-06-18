
find_package(rsa_mpi REQUIRED)
message( STATUS "rsa_mpi is used, header location: ${rsa_mpi_INCLUDE_DIRS}")
message( STATUS "rsa_mpi is used, rsa_mpi_DIR: ${rsa_mpi_DIR}")
include_directories(${rsa_mpi_INCLUDE_DIRS})

set(exadem_rsa_LINK_LIBRARIES exanbIO exanbDefBox exanbParticleNeighbors exadem_numerical_scheme exadem_friction exadem_force_field)
xstamp_add_plugin(exadem_rsa ${CMAKE_CURRENT_SOURCE_DIR})
