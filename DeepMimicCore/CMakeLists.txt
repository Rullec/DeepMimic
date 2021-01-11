cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)
set(CMAKE_CXX_STANDARD 17)

# fxd
execute_process(
  COMMAND python -c "from distutils.sysconfig import get_python_inc;print(get_python_inc())"
  OUTPUT_VARIABLE PYTHON_INC
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND python -c "import distutils.sysconfig as sysconfig;print(sysconfig.get_config_var('LIBDIR'))"
  OUTPUT_VARIABLE PYTHON_LIB
  OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "python inc path: ${PYTHON_INC}")
message(STATUS "python lib path: ${PYTHON_LIB}") 

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})

set(BULLTER_INC /home/xudong/Projects/bullet3/src)
if(APPLE)
    set(PYTHON_INC /Users/xudong/anaconda3/envs/spinningup/include/python3.6m)
    set(PYTHON_LIB /Users/xudong/anaconda3/envs/spinningup/lib -lpython3.6m)
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
add_definitions(-DBT_USE_DOUBLE_PRECISION -std=c++17 -march=native)

set(include_dir ${PYTHON_INC} ${BULLTER_INC} ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${include_dir})

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Build type: debug")
    add_definitions("-O0 -g -Wall -fPIC -ggdb ")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    message(STATUS "Build type: release")
    add_definitions("-O3 -g -fPIC")
else()
    message(STATUS "Build type: release(default)")
    add_definitions("-O3 -fPIC")
endif()
add_definitions("-Wno-inconsistent-missing-override -Wno-overloaded-virtual -Wno-unused-variable -Wno-reorder -Wno-int-to-pointer-cast -Wno-sign-compare -Wno-deprecated-declarations -Wno-unused-value -Werror=return-type")

add_subdirectory(util/json)
add_subdirectory(util)
add_subdirectory(anim)
add_subdirectory(render)
add_subdirectory(scenes)
add_subdirectory(sim)



set(libs ${OPENGL_LIBRARIES} GLEW  BulletCollision LinearMath Bullet3Common BulletInverseDynamics BulletInverseDynamicsUtils BulletDynamics BulletGenDynamics stdc++)
if(APPLE)
    set(libs ${libs} ${GLUT_LIBRARY} )
else()
    set(libs ${libs} glut)
endif()
set(mylibs SCENES SIM JSON ANIM RENDER )

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})


find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PYTHON_INC})


set(CMAKE_SWIG_FLAGS "")
if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-flat_namespace -undefined suppress")
endif()
# set up target library
set_source_files_properties(DeepMimicCore.i PROPERTIES CPLUSPLUS ON SWING_MODULE_NAME DeepMimicCore)
swig_add_library(DeepMimicCore TYPE SHARED LANGUAGE python OUTPUT_DIR ${CMAKE_SOURCE_DIR} OUTFILE_DIR ${CMAKE_SOURCE_DIR} SOURCES DeepMimicCore.i DeepMimicCore.cpp)
swig_link_libraries(DeepMimicCore ${mylibs} ${libs} ${PYTHON_LIB})

# set up target Main 
add_executable(Main Main.cpp DeepMimicCore.cpp DeepMimicCore)
target_link_libraries(Main ${mylibs} ${libs})
add_executable(Main_nodraw Main_nodraw.cpp DeepMimicCore.cpp DeepMimicCore)
target_link_libraries(Main_nodraw ${mylibs} ${libs})

# move main to its parent folder
add_custom_command(
    TARGET Main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rename
        ${CMAKE_CURRENT_BINARY_DIR}/Main        
        ${CMAKE_SOURCE_DIR}/Main    
)
add_custom_command(
    TARGET Main_nodraw POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rename
        ${CMAKE_CURRENT_BINARY_DIR}/Main_nodraw
        ${CMAKE_SOURCE_DIR}/Main_nodraw    
)

if(UNIX)
    add_custom_command(
        TARGET ${SWIG_MODULE_DeepMimicCore_REAL_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rename
            ${CMAKE_CURRENT_BINARY_DIR}/_DeepMimicCore.so
            ${CMAKE_SOURCE_DIR}/_DeepMimicCore.so
    )
    add_custom_command(
        TARGET ${SWIG_MODULE_DeepMimicCore_REAL_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rename
            ${CMAKE_CURRENT_BINARY_DIR}/DeepMimicCore.py
            ${CMAKE_SOURCE_DIR}/DeepMimicCore.py
    )
endif()