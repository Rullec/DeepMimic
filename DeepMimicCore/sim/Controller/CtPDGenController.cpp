#include "CtPDGenController.h"
#include "sim/SimItems/SimCharacterBase.h"
#include "sim/SimItems/SimJoint.h"
#include "util/LogUtil.hpp"

cCtPDGenController::cCtPDGenController()
{
    mGravity.setZero();
    mCurAction.resize(0);
    mCurPDTargetPose.resize(0);
}
cCtPDGenController::~cCtPDGenController() {}

void cCtPDGenController::Reset() { cCtController::Reset(); }
void cCtPDGenController::Clear() { cCtController::Clear(); }

void cCtPDGenController::SetGravity(const tVector &g) { mGravity = g; }

std::string cCtPDGenController::GetName() const { return "ct_pd_gen"; }

void cCtPDGenController::BuildStateOffsetScale(Eigen::VectorXd &out_offset,
                                               Eigen::VectorXd &out_scale) const
{
    MIMIC_WARN("hasn't been implemented");
}
void cCtPDGenController::BuildActionBounds(Eigen::VectorXd &out_min,
                                           Eigen::VectorXd &out_max) const
{
    MIMIC_WARN("hasn't been implemented");
}
void cCtPDGenController::BuildActionOffsetScale(
    Eigen::VectorXd &out_offset, Eigen::VectorXd &out_scale) const
{
    MIMIC_WARN("hasn't been implemented");
}
const tVectorXd &cCtPDGenController::GetCurAction() { return mCurAction; }

bool cCtPDGenController::ParseParams(const Json::Value &json)
{
    bool succ = cCtController::ParseParams(json);
    return succ;
}

void cCtPDGenController::UpdateBuildTau(double time_step,
                                        Eigen::VectorXd &out_tau)
{
    UpdatePDCtrls(time_step, out_tau);
}
void cCtPDGenController::SetupPDControllers(const Json::Value &json,
                                            const tVector &gravity)
{
    MIMIC_WARN("wait to be implemented\n");
    mValid = true;
}

/**
 * \brief                   Compute control torques (internal)
 * \param time_step
 * \param out_tau           output control torques
 *          If a valid PD target has been set to the PD controller,
 *  this function can calculate the control torque
 */
void cCtPDGenController::UpdatePDCtrls(double time_step,
                                       Eigen::VectorXd &out_tau)
{
    int num_of_dof = mChar->GetNumDof();
    out_tau.resize(num_of_dof);
    out_tau.setZero();
}

/**
 * \brief                   set the action(PD target) from outside
 * \param action
 */
void cCtPDGenController::ApplyAction(const Eigen::VectorXd &action) {}

/**
 * \brief                   build the action bounds (the action should not be
 * unlimited) \param joint_id \param out_min           lower bound of the action
 * \param out_max           upper bound of the action
 */
void cCtPDGenController::BuildJointActionBounds(int joint_id,
                                                Eigen::VectorXd &out_min,
                                                Eigen::VectorXd &out_max) const
{
    MIMIC_WARN("BuildJointActionBounds hasn't been finished");
}

/**
 * \brief
 */
void cCtPDGenController::BuildJointActionOffsetScale(
    int joint_id, Eigen::VectorXd &out_offset, Eigen::VectorXd &out_scale) const
{
    MIMIC_WARN("BuildJointActionOffsetScale hasn't been finished");
}
void cCtPDGenController::ConvertActionToTargetPose(
    int joint_id, Eigen::VectorXd &out_theta) const
{
    MIMIC_WARN("ConvertActionToTargetPose hasn't been finished");
}
void cCtPDGenController::ConvertTargetPoseToAction(
    int joint_id, Eigen::VectorXd &out_theta) const
{
    MIMIC_WARN("ConvertTargetPoseToAction hasn't been finished");
}

cKinTree::eJointType cCtPDGenController::GetJointType(int joint_id) const
{
    return mChar->GetJoint(joint_id).GetType();
}

void cCtPDGenController::SetPDTargets(const Eigen::VectorXd &targets)
{
    MIMIC_WARN("SetPDTargets hasn't been finished");
}