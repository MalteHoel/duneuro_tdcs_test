add_executable("duneuro_tdcs_fitted_test" duneuro_tdcs_fitted_test.cc)
target_link_dune_default_libraries("duneuro_tdcs_fitted_test")

add_executable("duneuro_tdcs_unfitted_test" duneuro_tdcs_unfitted_test.cc)
target_link_dune_default_libraries("duneuro_tdcs_unfitted_test")

# add symlinks to example data
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/fitted_configs.ini" "${CMAKE_CURRENT_BINARY_DIR}/fitted_configs.ini")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/mesh.msh" "${CMAKE_CURRENT_BINARY_DIR}/mesh.msh")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/conductivities.txt" "${CMAKE_CURRENT_BINARY_DIR}/conductivities.txt")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/electrodes.txt" "${CMAKE_CURRENT_BINARY_DIR}/electrodes.txt")

execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/unfitted_configs.ini" "${CMAKE_CURRENT_BINARY_DIR}/unfitted_configs.ini")
