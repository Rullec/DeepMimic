#include "SimbiconController.h"
#include "sim/SimItems/SimCharacterBase.h"
#include "util/JsonUtil.h"
#include "util/LogUtil.h"
#include <iostream>

extern const std::string gJointTypeNames[cKinTree::eJointTypeMax];
extern const std::string gPDControllersKey;
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

    // 3. debug test set target random pose
    int num_of_dof = mChar->GetNumDof();
    tVectorXd tar_pose = tVectorXd::Random(num_of_dof),
              tar_vel = tVectorXd::Zero(num_of_dof);
    SetTargetTheta(tar_pose);
    SetTargetVel(tar_vel);
}

/**
 * \brief           Init finite state machine
*/
void cSimbiconController::InitFSM(const Json::Value &conf)
{
    MIMIC_INFO("fsm begin to init, hasn't been implemented");
}

/**
 * \brief           Given current time step, calculate the SIMBICON control force
 * 
 * this function will be called in updatecharacter and DeepMimicController::Update()
 * the format of out_tau should be the same as the character->applycontrolforce()
*/
void cSimbiconController::UpdateCalcTau(double timestep,
                                        Eigen::VectorXd &out_tau)
{

    // the custom UpdateBuildTau will be called in this function
    cCtPDController::UpdateCalcTau(timestep, out_tau);
}

/**
 * \brief           Calculate the control force (indeed this time)
 * It will be called by UpdateCalcTau
*/
void cSimbiconController::UpdateBuildTau(double time_step, tVectorXd &out_tau)
{
    // 1. set the target theta

    // 2. calculate the PD control force
    mPDCtrl.UpdateControlForce(time_step, out_tau);
}

/**
 * 
*/
void cSimbiconController::Reset() { cCtPDController::Reset(); }
void cSimbiconController::Clear() { cCtPDController::Clear(); }

const tVectorXd &cSimbiconController::GetCurAction() const
{
    return this->mAction;
}
const tVectorXd &cSimbiconController::GetCurPDTargetPose() const
{
    MIMIC_ERROR("hasn't been implemented");
    return tVectorXd::Zero(0);
}
void cSimbiconController::CalcActionByTargetPose(tVectorXd &pd_target)
{
    MIMIC_ERROR("hasn't been implemented");
}
void cSimbiconController::CalcPDTargetByTorque(double dt, const tVectorXd &pose,
                                               const tVectorXd &vel,
                                               const tVectorXd &torque,
                                               tVectorXd &pd_target)
{
    MIMIC_ERROR("hasn't been implemented");
}

std::string cSimbiconController::GetName() const { return "simbicon_ctrl"; }

/**
 * \brief           Give the full size target pose, set it to another place
*/
void cSimbiconController::SetTargetTheta(const tVectorXd &tar_pose)
{
    int num_of_dof = mChar->GetNumDof();
    MIMIC_ASSERT(tar_pose.size() == num_of_dof);

    int num_of_joint = this->mChar->GetNumJoints();
    for (int i = 0; i < num_of_joint; i++)
    {
        // set the PD target for each joint, ignore root
        if (i != mChar->GetRootID())
        {
            cPDController &ctrl = mPDCtrl.GetPDCtrl(i);
            const cSimBodyJoint &joint = mChar->GetJoint(i);
            int offset = mChar->GetParamOffset(i),
                size = mChar->GetParamSize(i);

            // target pose
            mPDCtrl.SetTargetTheta(i, tar_pose.segment(offset, size));
        }
    }
}

/**
 * \brief           Set fullsize target vel
*/
void cSimbiconController::SetTargetVel(const tVectorXd &tar_vel)
{
    int num_of_dof = mChar->GetNumDof();
    MIMIC_ASSERT(tar_vel.size() == num_of_dof);

    int num_of_joint = this->mChar->GetNumJoints();
    for (int i = 0; i < num_of_joint; i++)
    {
        // set the PD target vel for each joint, ignore root
        if (i != mChar->GetRootID())
        {
            cPDController &ctrl = mPDCtrl.GetPDCtrl(i);
            const cSimBodyJoint &joint = mChar->GetJoint(i);
            int offset = mChar->GetParamOffset(i),
                size = mChar->GetParamSize(i);

            // target vel
            mPDCtrl.SetTargetVel(i, tar_vel.segment(offset, size));
        }
    }
}