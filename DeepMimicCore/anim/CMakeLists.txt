cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)

set(DIR_SRCS
Character.cpp
KinCharacter.cpp
KinTree.cpp
Motion.cpp
Shape.cpp
)
add_library(ANIM ${DIR_SRCS})
target_link_libraries(ANIM UTIL)