enable_testing()

find_package(OpenMP REQUIRED)

list(APPEND TLIST compute_hooke gravity init_fields)

foreach(varTest IN LISTS TLIST)
	add_executable(demTest_${varTest} test_${varTest}.cpp)
	target_compile_options(demTest_${varTest} PRIVATE ${OpenMP_CXX_FLAGS})
	target_include_directories(demTest_${varTest} PUBLIC ${USTAMP_INCLUDE_DIRS})
	target_link_libraries(demTest_${varTest} PUBLIC cea_des_dem exanbCompute ${USTAMP_CORE_LIBRARIES} ${OpenMP_CXX_LIBRARIES} dl)
	add_test(basicDem${varTest} demTest_${varTest})
	install(TARGETS demTest_${varTest} DESTINATION bin)
endforeach()


