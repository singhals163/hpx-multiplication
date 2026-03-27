cmake_minimum_required(VERSION 3.18)
project(HPX_MatrixMultiplication CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(HPX REQUIRED)

add_executable(hpx_matmul multiplication.cpp)

target_link_libraries(hpx_matmul PUBLIC HPX::hpx HPX::wrap_main)