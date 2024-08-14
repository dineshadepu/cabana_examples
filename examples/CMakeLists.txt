add_executable(02 02_parallel_for.cpp)
target_link_libraries(02 LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

add_executable(03 03_compute_local_quantity_in_parallel_func.cpp)
target_link_libraries(03 LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

add_executable(05_aosoa_copy_host_back_mirror 05_aosoa_copy_host_back_mirror.cpp)
target_link_libraries(05_aosoa_copy_host_back_mirror LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

add_executable(function_approximation function_approximation.cpp)
target_link_libraries(function_approximation LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

install(TARGETS function_approximation 02 03 DESTINATION ${CMAKE_INSTALL_BINDIR})
