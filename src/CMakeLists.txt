# ===================================
# === exaDEM application package  ===
# ===================================

add_subdirectory(grid)
add_subdirectory(fieldmutator)
add_subdirectory(numericalscheme)
add_subdirectory(friction)
add_subdirectory(forcefield)
add_subdirectory(io)
add_subdirectory(driver)
add_subdirectory(meshset)
add_subdirectory(polyhedra)

if(USE_RSA)
	add_subdirectory(rsa)
endif()

AddRegressionTestDir(${CMAKE_CURRENT_SOURCE_DIR}/regression)

 
# automatically added by xstamp_add_plugin macro
#add_subdirectory(tests)

