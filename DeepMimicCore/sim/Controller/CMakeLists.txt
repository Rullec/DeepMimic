set(CONTROLLER_SRC
    CharController.cpp
    Controller.cpp
    CtController.cpp
    CtCtrlUtil.cpp
    CtPDController.cpp
    CtrlBuilder.cpp
    CtVelController.cpp
    DeepMimicCharController.cpp
    ExpPDController.cpp
    ImpPDController.cpp
    PDController.cpp
    RBDModel.cpp
    RBDUtil.cpp
    ReverseController.cpp
)
add_library(CONTROLLER_LIB ${CONTROLLER_SRC})