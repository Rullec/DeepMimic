cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)
set(CMAKE_CXX_STANDARD 11)

set(PYTHON_INC /home/ljf/anaconda3/include/python3.7m)
set(PYTHON_LIB /home/ljf/anaconda3/lib/libpython3.7m.so)
set(EIGEN_INC /usr/local/include/eigen3)
set(BULLTER_INC /usr/local/include/bullet)

set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})

set(include_dir ${PYTHON_INC} ${EIGEN_INC} ${BULLTER_INC} ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${include_dir})

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions("-O0 -Wall -fPIC -ggdb")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    add_definitions("-O3 -Wall -fPIC")
endif()

add_subdirectory(anim)
add_subdirectory(render)
add_subdirectory(scenes)
add_subdirectory(sim)
add_subdirectory(util)
add_subdirectory(util/json)

set(libs ${OPENGL_LIBRARIES} GLEW glut BulletDynamics BulletCollision LinearMath m stdc++)
set(mylibs SCENES SIM JSON ANIM RENDER UTIL )

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PYTHON_INC})

set(CMAKE_SWIG_FLAGS "")
set_source_files_properties(DeepMimicCore.i PROPERTIES CPLUSPLUS ON)

swig_add_library(DeepMimicCore LANGUAGE python OUTPUT_DIR ${CMAKE_SOURCE_DIR} OUTFILE_DIR ${CMAKE_SOURCE_DIR} SOURCES DeepMimicCore.i DeepMimicCore.cpp)

swig_link_libraries(DeepMimicCore ${mylibs} ${libs} ${PYTHON_LIBRARIES})

INSTALL(FILES cmake-build-release/_DeepMimicCore.so
        DESTINATION ${CMAKE_SOURCE_DIR})