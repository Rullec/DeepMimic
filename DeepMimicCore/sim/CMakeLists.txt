cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)


find_package(MPI REQUIRED)
if(MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif()

aux_source_directory(. DIR_SRCS)
add_library(SIM ${DIR_SRCS} )
target_link_libraries(SIM ${MPI_LIBRARIES})