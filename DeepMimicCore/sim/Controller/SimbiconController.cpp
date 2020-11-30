#include "SimbiconController.h"
#include "util/JsonUtil.h"
#include "util/LogUtil.h"
#include <iostream>

cSimbiconController::cSimbiconController() {}
cSimbiconController::~cSimbiconController() {}

void cSimbiconController::Init(cSimCharacterBase *character,
                               const std::string &ctrl_file)
{
    cDeepMimicCharController::Init(character, ctrl_file);

    // 1. init the PD controller
    Json::Value root;
    cJsonUtil::LoadJson(ctrl_file, root);
    tMatrixXd pd_params;
    cPDController::LoadParams(cJsonUtil::ParseAsValue(gPDControllersKey, root),
                              pd_params);
    mPDCtrl.Init(character, pd_params);

    // 2. init the FSM
    InitFSM(root);
}

/**
 * \brief           Init finite state machine
*/
void cSimbiconController::InitFSM(const Json::Value &conf)
{
    MIMIC_INFO("fsm begin to init");
}

std::string cSimbiconController::GetName() const { return "simbicon_ctrl"; }