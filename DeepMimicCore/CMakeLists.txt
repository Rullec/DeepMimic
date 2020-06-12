cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)
set(CMAKE_CXX_STANDARD 17)

# please use soft linkage to set your python headers & libs
set(PYTHON_INC ./include/include/python3.6m)
set(PYTHON_LIB_PATH ./include/lib)
set(PYTHON_LIB libpython3.6m.so)
link_directories(${PYTHON_LIB_PATH})
include_directories(${PYTHON_INC})

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

# find opengl packages
set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
add_definitions(-DBT_USE_DOUBLE_PRECISION  -std=c++17 -march=native)

# find eigen packages
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
set(include_dir ${PYTHON_INC} ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${include_dir})

# find bullet packages
find_package(Bullet REQUIRED)
if (Bullet_FOUND)
    message(STATUS ${BULLET_INCLUDE_DIRS})
    include_directories(${BULLET_INCLUDE_DIRS})
    message(STATUS ${BULLET_LIBRARIES})
endif()

#different build type
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
#add_definitions("-Wno-overloaded-virtual -Wno-unused-variable -Wno-reorder -Wno-int-to-pointer-cast -Wno-unused-but-set-variable -Wno-sign-compare -Wno-deprecated-declarations -Wno-unused-value -Werror=return-type")
add_definitions("-Wno-overloaded-virtual -Wno-unused-variable -Wno-reorder -Wno-int-to-pointer-cast -Wno-unused-const-variable -Wno-sign-compare -Wno-deprecated-declarations -Wno-unused-value -Werror=return-type")

add_subdirectory(util/json)
add_subdirectory(util)
add_subdirectory(anim)
add_subdirectory(render)
add_subdirectory(scenes)
add_subdirectory(sim)

set(libs ${OPENGL_LIBRARIES} ${BULLET_LIBRARIES} GLEW BulletCollision LinearMath Bullet3Common BulletInverseDynamics BulletInverseDynamicsUtils BulletDynamics stdc++)
if(APPLE)
    set(libs ${libs} ${GLUT_LIBRARY} )
else()
    set(libs ${libs} glut)
endif()
set(mylibs SCENES SIM JSON ANIM RENDER )

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})


set(CMAKE_SWIG_FLAGS "")
if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-flat_namespace -undefined suppress")
endif()
# set up target library
set_source_files_properties(DeepMimicCore.i PROPERTIES CPLUSPLUS ON)
swig_add_library(DeepMimicCore TYPE SHARED LANGUAGE python OUTPUT_DIR ${CMAKE_SOURCE_DIR} OUTFILE_DIR ${CMAKE_SOURCE_DIR} SOURCES DeepMimicCore.i DeepMimicCore.cpp)
swig_link_libraries(DeepMimicCore ${mylibs} ${libs} ${PYTHON_LIB})

# set up target Main 
add_executable(Main Main.cpp DeepMimicCore.cpp DeepMimicCore)
target_link_libraries(Main ${mylibs} ${libs})

# move main to its parent folder
if(EXISTS "../Main")
    add_custom_command(TARGET Main
        PRE_BUILD
        COMMAND rm ../Main
        )
endif()
add_custom_command(TARGET Main
    POST_BUILD
    COMMAND mv ./Main ../Main
    )

# move the .dylib in build folder to .so in the parent folder
if(UNIX)
    if(EXISTS "../_DeepMimicCore.so")
        add_custom_command(TARGET Main
            PRE_BUILD
            COMMAND rm ../_DeepMimicCore.so
            )
    endif()
    add_custom_command(TARGET Main
        POST_BUILD
        COMMAND mv ./_DeepMimicCore.* ../_DeepMimicCore.so
        )
endif()
