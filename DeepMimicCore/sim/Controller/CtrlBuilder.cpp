#include "sim/Controller/CtrlBuilder.h"
#include "sim/Controller/CtController.h"
#include "sim/Controller/CtPDController.h"
#include "sim/Controller/CtPDGenController.h"
#include "sim/Controller/CtVelController.h"
#include "util/LogUtil.h"
#include <iostream>
using namespace std;

const std::string gCharCtrlName[cCtrlBuilder::eCharCtrlMax] = {
    "none", "ct", "ct_pd", "ct_pd_gen", "ct_vel"};

cCtrlBuilder::tCtrlParams::tCtrlParams()
{
    mCharCtrl = eCharCtrlNone;
    mCtrlFile = "";

    mChar = nullptr;
    mGround = nullptr;
    mGravity = gGravity;
}

void cCtrlBuilder::ParseCharCtrl(const std::string &char_ctrl_str,
                                 eCharCtrl &out_char_ctrl)
{
    bool found = false;
    if (char_ctrl_str == "" || char_ctrl_str == "none")
    {
        out_char_ctrl = eCharCtrlNone;
        found = true;
    }
    else
    {
        for (int i = 0; i < eCharCtrlMax; ++i)
        {
            const std::string &name = gCharCtrlName[i];
            if (char_ctrl_str == name)
            {
                out_char_ctrl = static_cast<eCharCtrl>(i);
                found = true;
                break;
            }
        }
    }

    if (!found)
    {
        assert(false &&
               "Unsupported character controller\n"); // unsupported character
                                                      // controller
    }
}

bool cCtrlBuilder::BuildController(const tCtrlParams &params,
                                   std::shared_ptr<cCharController> &out_ctrl)
{
    bool succ = true;
    MIMIC_INFO("build controller type {}", gCharCtrlStr[params.mCharCtrl]);
    switch (params.mCharCtrl)
    {
    case eCharCtrlNone:
        break;
    case eCharCtrlCt:
        succ = BuildCtController(params, out_ctrl);
        break;
    case eCharCtrlCtPD:
        succ = BuildCtPDController(params, out_ctrl);
        break;
    case eCharCtrlCtPDGen:
        succ = BuildCtPDGenController(params, out_ctrl);
        break;
    case eCharCtrlCtVel:
        succ = BuildCtVelController(params, out_ctrl);
        break;
    default:
        assert(false &&
               "Failed Building Unsupported Controller\n"); // unsupported
                                                            // controller
        break;
    }

    return succ;
}

bool cCtrlBuilder::BuildCtController(const tCtrlParams &params,
                                     std::shared_ptr<cCharController> &out_ctrl)
{
    bool succ = true;
    std::shared_ptr<cCtController> ctrl =
        std::shared_ptr<cCtController>(new cCtController());
    ctrl->Init(params.mChar.get(), params.mCtrlFile);

    out_ctrl = ctrl;
    return succ;
}

bool cCtrlBuilder::BuildCtPDController(
    const tCtrlParams &params, std::shared_ptr<cCharController> &out_ctrl)
{
    bool succ = true;
    std::shared_ptr<cCtPDController> ctrl =
        std::shared_ptr<cCtPDController>(new cCtPDController());
    ctrl->SetGravity(params.mGravity);
    ctrl->Init(params.mChar.get(), params.mCtrlFile);

    out_ctrl = ctrl;
    return succ;
}

bool cCtrlBuilder::BuildCtPDGenController(
    const tCtrlParams &params, std::shared_ptr<cCharController> &out_ctrl)
{
    bool succ = true;
    std::shared_ptr<cCtPDGenController> ctrl =
        std::shared_ptr<cCtPDGenController>(new cCtPDGenController());
    ctrl->SetGravity(params.mGravity);
    ctrl->Init(params.mChar.get(), params.mCtrlFile);

    out_ctrl = ctrl;
    return succ;
}

bool cCtrlBuilder::BuildCtVelController(
    const tCtrlParams &params, std::shared_ptr<cCharController> &out_ctrl)
{
    bool succ = true;
    std::shared_ptr<cCtVelController> ctrl =
        std::shared_ptr<cCtVelController>(new cCtVelController());
    ctrl->SetGravity(params.mGravity);
    ctrl->Init(params.mChar.get(), params.mCtrlFile);

    out_ctrl = ctrl;
    return succ;
}