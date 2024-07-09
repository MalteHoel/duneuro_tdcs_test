add_executable("duneuro_tdcs_test" duneuro_tdcs_test.cc)
target_link_dune_default_libraries("duneuro_tdcs_test")

# add symlinks to example data
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/configs.ini" "${CMAKE_CURRENT_BINARY_DIR}/configs.ini")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/mesh.msh" "${CMAKE_CURRENT_BINARY_DIR}/mesh.msh")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/conductivities.txt" "${CMAKE_CURRENT_BINARY_DIR}/conductivities.txt")
execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/input/electrodes.txt" "${CMAKE_CURRENT_BINARY_DIR}/electrodes.txt")
