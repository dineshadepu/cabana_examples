add_executable(02 02_parallel_for.cpp)
target_link_libraries(02 LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

add_executable(03 03_compute_local_quantity_in_parallel_func.cpp)
target_link_libraries(03 LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

install(TARGETS 02 03 DESTINATION ${CMAKE_INSTALL_BINDIR})
