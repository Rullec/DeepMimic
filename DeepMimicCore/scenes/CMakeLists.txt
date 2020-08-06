cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)

set(DIR_SRCS
DrawRLScene.cpp
DrawScene.cpp
DrawSceneImitate.cpp
DrawSceneKinChar.cpp
DrawSceneSimChar.cpp
RLScene.cpp
RLSceneSimChar.cpp
SceneBuilder.cpp
Scene.cpp
SceneImitate.cpp
SceneKinChar.cpp
SceneSimChar.cpp
)
add_library(SCENES ${DIR_SRCS} )
target_link_libraries(SCENES UTIL SIM)