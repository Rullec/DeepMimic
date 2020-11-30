#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sim/Controller/DeepMimicCharController.h"
#include "sim/SimItems/SimCharacter.h"
#include "sim/World/Ground.h"

class cCtrlBuilder
{
public:
    enum eCharCtrl
    {
        eCharCtrlNone,
        eCharCtrlCt,
        eCharCtrlCtPD,
        eCharCtrlCtPDGen,
        eCharCtrlCtVel,
        eCharctrlSimbicon,
        eCharCtrlMax,
        NUM_CTRL_TYPE
    };
    struct tCtrlParams
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        eCharCtrl mCharCtrl;
        std::string mCtrlFile;

        std::shared_ptr<cSimCharacterBase> mChar;
        std::shared_ptr<cGround> mGround;
        tVector mGravity;

        tCtrlParams();
    };

    static void ParseCharCtrl(const std::string &char_ctrl_str,
                              eCharCtrl &out_char_ctrl);
    static bool BuildController(const tCtrlParams &params,
                                std::shared_ptr<cCharController> &out_ctrl);

protected:
    static bool BuildCtController(const tCtrlParams &params,
                                  std::shared_ptr<cCharController> &out_ctrl);
    static bool BuildCtPDController(const tCtrlParams &params,
                                    std::shared_ptr<cCharController> &out_ctrl);
    static bool
    BuildCtPDGenController(const tCtrlParams &params,
                           std::shared_ptr<cCharController> &out_ctrl);
    static bool
    BuildCtVelController(const tCtrlParams &params,
                         std::shared_ptr<cCharController> &out_ctrl);
    static bool
    BuildCtTargetController(const tCtrlParams &params,
                            std::shared_ptr<cCharController> &out_ctrl);

    static bool BuildDogController(const tCtrlParams &params,
                                   std::shared_ptr<cCharController> &out_ctrl);
    static bool
    BuildRaptorController(const tCtrlParams &params,
                          std::shared_ptr<cCharController> &out_ctrl);

    static bool
    BuildBiped3DStepController(const tCtrlParams &params,
                               std::shared_ptr<cCharController> &out_ctrl);
    static bool
    BuildBiped3DSymStepController(const tCtrlParams &params,
                                  std::shared_ptr<cCharController> &out_ctrl);

    static bool
    BuildCtTarPoseController(const tCtrlParams &params,
                             std::shared_ptr<cCharController> &out_ctrl);
    static bool
    BuildCtSymPDController(const tCtrlParams &params,
                           std::shared_ptr<cCharController> &out_ctrl);

    static bool
    BuildCtHeadingController(const tCtrlParams &params,
                             std::shared_ptr<cCharController> &out_ctrl);
    static bool
    BuildCtHeadingPDController(const tCtrlParams &params,
                               std::shared_ptr<cCharController> &out_ctrl);
    static bool
    BuildCtSymHeadingPDController(const tCtrlParams &params,
                                  std::shared_ptr<cCharController> &out_ctrl);
    static bool
    BuildCtStrikePDController(const tCtrlParams &params,
                              std::shared_ptr<cCharController> &out_ctrl);
    static bool
    BuildCtStrikeObjPDController(const tCtrlParams &params,
                                 std::shared_ptr<cCharController> &out_ctrl);
    static bool
    BuildCtCmdPDController(const tCtrlParams &params,
                           std::shared_ptr<cCharController> &out_ctrl);
    static bool
    BuildSimbiconController(const tCtrlParams &params,
                            std::shared_ptr<cCharController> &out_ctrl);
};

const std::string gCharCtrlStr[cCtrlBuilder::eCharCtrl::NUM_CTRL_TYPE] = {
    "eCharCtrlNone",    "eCharCtrlCt",    "eCharCtrlCtPD",
    "eCharCtrlCtPDGen", "eCharCtrlCtVel", "eCharCtrlMax"};