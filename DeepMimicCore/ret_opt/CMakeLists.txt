cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)

aux_source_directory(. DIR_SRCS)
add_library(DMRETOPT ${DIR_SRCS})

