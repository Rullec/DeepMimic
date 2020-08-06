cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)

add_definitions("-O3 -Wall -fPIC")
set(DIR_SRCS
json_reader.cpp
json_value.cpp
json_writer.cpp
)
add_library(JSON ${DIR_SRCS})