cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)
set(CMAKE_CXX_STANDARD 11)

# ljf
# set(PYTHON_INC /home/ljf/anaconda3/include/python3.7m)
# set(PYTHON_LIB /home/ljf/anaconda3/lib/libpython3.7m.so)
# set(EIGEN_INC /usr/local/include/eigen3)
# set(BULLTER_INC /usr/local/include/bullet)

# fxd
set(PYTHON_INC /home/xudong/.conda/envs/spinningup/include/python3.6m)
set(PYTHON_LIB /home/xudong/.conda/envs/spinningup/lib -lpython3.6m)
set(EIGEN_INC /home/xudong/Projects/include/)
set(BULLTER_INC /home/xudong/Projects/bullet3/src)
if(APPLE)
    set(PYTHON_INC /Users/xudong/anaconda3/envs/spinningup/include/python3.6m)
    set(PYTHON_LIB /Users/xudong/anaconda3/envs/spinningup/lib -lpython3.6m)
    set(EIGEN_INC /Users/xudong/Projects/include/)
    set(BULLTER_INC /Users/xudong/Projects/bullet3/src)
    # find_package(GLEW REQUIRED)
    find_package(PkgConfig REQUIRED)

    # glew
    pkg_search_module(GLEW REQUIRED glew)
    if(NOT GLEW_FOUND)
    message(ERROR " GLEW not found!")
    endif()
    include_directories(${GLEW_INCLUDE_DIRS})
    link_directories(${GLEW_LIBRARY_DIRS})

    # glut
    FIND_LIBRARY(GLUT_LIBRARY GLUT )
    link_directories(/Users/xudong/Projects/bullet3/build_cmake/src/BulletDynamics)
    link_directories(/Users/xudong/Projects/bullet3/build_cmake/src/BulletCollision)
    link_directories(/Users/xudong/Projects/bullet3/build_cmake/src/LinearMath)
    link_directories(/Users/xudong/Projects/bullet3/build_cmake/src/Bullet3Collision)
    link_directories(/Users/xudong/Projects/bullet3/build_cmake/src/Bullet3Common)
    link_directories(/Users/xudong/Projects/bullet3/build_cmake/src/BulletInverseDynamics)
    link_directories(/Users/xudong/Projects/bullet3/build_cmake/Extras/InverseDynamics)
endif()


set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
add_definitions( -DBT_USE_DOUBLE_PRECISION )

set(include_dir ${PYTHON_INC} ${EIGEN_INC} ${BULLTER_INC} ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${include_dir})

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Build type: debug")
    add_definitions("-O0 -Wall -fPIC -ggdb ")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    message(STATUS "Build type: release")
    add_definitions("-O3 -fPIC")
else()
    message(STATUS "Build type: release(default)")
    add_definitions("-O3 -fPIC")
endif()
add_definitions("-Wno-overloaded-virtual -Wno-unused-variable -Wno-reorder -Wno-int-to-void-pointer-cast -Wno-deprecated-declarations -Wno-unused-value -Werror=return-type")

add_subdirectory(anim)
add_subdirectory(render)
add_subdirectory(scenes)
add_subdirectory(sim)
add_subdirectory(util)
add_subdirectory(util/json)
if(APPLE)
    set(libs ${OPENGL_LIBRARIES} GLEW ${GLUT_LIBRARY} BulletDynamics BulletCollision LinearMath Bullet3Common BulletInverseDynamics BulletInverseDynamicsUtils stdc++)
else()
    set(libs ${OPENGL_LIBRARIES} GLEW glut BulletDynamics BulletCollision LinearMath Bullet3Common BulletInverseDynamics BulletInverseDynamicsUtils stdc++)
endif()
set(mylibs SCENES SIM JSON ANIM RENDER UTIL )

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PYTHON_INC})

set(CMAKE_SWIG_FLAGS "")
set_source_files_properties(DeepMimicCore.i PROPERTIES CPLUSPLUS ON)

swig_add_library(DeepMimicCore LANGUAGE python OUTPUT_DIR ${CMAKE_SOURCE_DIR} OUTFILE_DIR ${CMAKE_SOURCE_DIR} SOURCES DeepMimicCore.i DeepMimicCore.cpp)

# swig_link_libraries(DeepMimicCore ${mylibs} ${libs} ${PYTHON_LIBRARIES})
swig_link_libraries(DeepMimicCore ${mylibs} ${libs} ${PYTHON_LIB})

# cmake for Main 
add_executable(Main main.cpp DeepMimicCore.cpp)
target_link_libraries(Main ${mylibs} ${libs})

INSTALL(FILES cmake-build-release/_DeepMimicCore.so
        DESTINATION ${CMAKE_SOURCE_DIR})