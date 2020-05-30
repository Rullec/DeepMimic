cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)

aux_source_directory(. DIR_SRCS)
add_library(ANIM ${DIR_SRCS})
target_link_libraries(ANIM UTIL)