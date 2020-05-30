cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)

find_package(spdlog REQUIRED)
aux_source_directory(. DIR_SRCS)
add_library(UTIL ${DIR_SRCS})
target_link_libraries(UTIL spdlog::spdlog JSON stdc++fs)