cmake_minimum_required(VERSION 3.5)
project(Eigen34)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(SOURCES example.cpp)

add_executable(example ${SOURCES})
target_compile_options(example PRIVATE -Wall)
