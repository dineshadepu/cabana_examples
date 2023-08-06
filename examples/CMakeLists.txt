add_executable(02 02_parallel_for.cpp)
target_link_libraries(02 LINK_PUBLIC Cabana::cabanacore Cabana::Cajita)

install(TARGETS 02 DESTINATION ${CMAKE_INSTALL_BINDIR})
