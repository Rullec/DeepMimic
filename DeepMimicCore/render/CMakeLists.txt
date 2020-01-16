cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)

aux_source_directory(. DIR_SRCS)
add_library(RENDER ${DIR_SRCS} lodepng/lodepng.cpp lodepng/lodepng_util.cpp)