cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)

set(DIR_SRCS
Camera.cpp
DrawCharacter.cpp
DrawGround.cpp
DrawKinTree.cpp
DrawMesh.cpp
DrawObj.cpp
DrawPerturb.cpp
DrawSimCharacter.cpp
DrawUtil.cpp
DrawWorld.cpp
GraphUtil.cpp
IBuffer.cpp
MatrixStack.cpp
MeshUtil.cpp
Shader.cpp
ShadowMap.cpp
TextureDesc.cpp
TextureUtil.cpp
VertexBuffer.cpp
)
add_library(RENDER ${DIR_SRCS} lodepng/lodepng.cpp lodepng/lodepng_util.cpp)