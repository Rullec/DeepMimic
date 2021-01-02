cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)

find_package(MPI REQUIRED)
if(MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif()

find_package(spdlog REQUIRED)
set(DIR_SRCS
Annealer.cpp
ArgParser.cpp
BulletUtil.cpp
BVHReader.cpp
cTimeUtil.cpp
FileUtil.cpp
IndexManager.cpp
JsonUtil.cpp
LogUtil.cpp
MathUtil.cpp
Rand.cpp
Timer.cpp
TimeUtil.cpp
Trajectory.cpp
MPIUtil.cpp
)
add_library(UTIL ${DIR_SRCS})


target_link_libraries(UTIL spdlog::spdlog JSON ${MPI_LIBRARIES} )

if(Linux)
    target_link_libraries(UTIL stdc++fs)
endif()

