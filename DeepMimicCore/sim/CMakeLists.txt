cmake_minimum_required(VERSION 3.5)
project(DeepMimicCore)


find_package(MPI REQUIRED)
if(MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif()

set(CONTROLLER_SRC
    Controller/CharController.cpp
    Controller/Controller.cpp
    Controller/CtController.cpp
    Controller/CtCtrlUtil.cpp
    Controller/CtPDController.cpp
    Controller/CtPDGenController.cpp
    Controller/CtrlBuilder.cpp
    Controller/CtVelController.cpp
    Controller/DeepMimicCharController.cpp
    Controller/ExpPDController.cpp
    Controller/ImpPDController.cpp
    Controller/PDController.cpp
    Controller/RBDModel.cpp
    Controller/RBDUtil.cpp
    Controller/ReverseController.cpp
)

set(
    TrajManager_SRC
    TrajManager/BuildIDSolver.cpp
    TrajManager/DisplayIDSolver.cpp
    TrajManager/IDSolver.cpp
    TrajManager/InteractiveIDSolver.cpp
    TrajManager/OfflineSolveIDSolver.cpp
    TrajManager/OnlineIDSolver.cpp
    TrajManager/SampleIDSolver.cpp    
    TrajManager/TrajRecorder.cpp    
)

set(
    SIMITEMS_SRC
    SimItems/MultiBody.cpp
    SimItems/SimBodyJoint.cpp
    SimItems/SimBodyLink.cpp
    SimItems/SimBodyJointGen.cpp
    SimItems/SimBodyLinkGen.cpp
    SimItems/SimBox.cpp
    SimItems/SimCapsule.cpp
    SimItems/SimCharacterBase.cpp
    SimItems/SimCharacter.cpp
    SimItems/SimCharBuilder.cpp
    SimItems/SimCharGeneral.cpp
    SimItems/SimCylinder.cpp
    SimItems/SimJoint.cpp
    SimItems/SimObj.cpp
    SimItems/SimPlane.cpp
    SimItems/SimRigidBody.cpp
    SimItems/SimSphere.cpp
    SimItems/SpAlg.cpp
    SimItems/SimCharacterGen.cpp
)

set(
    WORLD_SRC
    World/AgentRegistry.cpp
    World/ContactManager.cpp
    World/GroundBuilder.cpp
    World/Ground.cpp
    World/GroundPlane.cpp
    World/ObjTracer.cpp
    World/Perturb.cpp
    World/PerturbManager.cpp
    World/FeaWorld.cpp
    World/GenWorld.cpp
    World/WorldBuilder.cpp
    World/WorldBase.cpp
)
set(DIR_SRCS ${CONTROLLER_SRC} ${TrajManager_SRC} ${SIMITEMS_SRC} ${WORLD_SRC})
add_library(SIM ${DIR_SRCS} )
target_link_libraries(SIM ${MPI_LIBRARIES} UTIL)