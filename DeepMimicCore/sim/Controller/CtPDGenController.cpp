#include "CtPDGenController.h"
#include "ImpPDGenController.h"
#include "sim/SimItems/SimCharacterBase.h"
#include "sim/SimItems/SimCharacterGen.h"
#include "sim/SimItems/SimJoint.h"
#include "sim/TrajManager/TrajRecorder.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include "util/LogUtil.h"
#include <iostream>

static std::string pd_log = "pd_log.txt";
cCtPDGenController::cCtPDGenController()
{
    mGravity.setZero();
    mCurAction.resize(0);
    mCurPDTargetPose.resize(0);
    mEnableGuidedAction = false;
    mGuidedTrajFile = "";
    mInternalFrameId = 0;
    mLoadInfo = nullptr;
    cFileUtil::ClearFile(pd_log);
}
cCtPDGenController::~cCtPDGenController()
{

    if (mLoadInfo != nullptr)
        delete mLoadInfo;
}

void cCtPDGenController::Init(cSimCharacterBase *character,
                              const std::string &param_file)
{
    cCtController::Init(character, param_file);

    // // load the parameter file again
    // MIMIC_INFO("PDGenController is initialized {}", param_file);
    // Json::Value root;
    // MIMIC_ASSERT(cJsonUtil::LoadJson(param_file, root));

    // mEnableLoadedAction = cJsonUtil::ParseAsBool("EnableLoadedAction", root);

    // if (mEnableLoadedAction)
    // {
    //     mGuidedTrajFile = cJsonUtil::ParseAsString("GuidedTrajFile", root);

    //     MIMIC_ASSERT(cFileUtil::ExistsFile(mGuidedTrajFile) &&
    //                  "the traj file doesn't exist ");
    //     MIMIC_INFO("ActionGuide enabled; traj file {}", mGuidedTrajFile);
    //     mLoadInfo = new tLoadInfo();
    // }
}

void cCtPDGenController::SetGuidedControlInfo(bool enable,
                                              const std::string &guide_file)
{
    MIMIC_INFO("set guided control info {} {}", enable, guide_file);
    if (enable == true)
    {
        if (mLoadInfo != nullptr)
            delete mLoadInfo;
        mLoadInfo = new tLoadInfo();
        mGuidedTrajFile = guide_file;
        mEnableGuidedAction = enable;
    }
}
void cCtPDGenController::Reset()
{
    cCtController::Reset();
    mInternalFrameId = 0;
}
void cCtPDGenController::Clear()
{
    cCtController::Clear();
    mInternalFrameId = 0;
}

void cCtPDGenController::SetGravity(const tVector &g) { mGravity = g; }

std::string cCtPDGenController::GetName() const { return "ct_pd_gen"; }

/**
 * \brief               Build the offset & scale in Gaussian distribution for
 * state
 *
 *          States are used as the input of the Neural network in order to
 * represent the current status of char. We should assume this state obey some
 * particular distribution, such as multivar Gaussian distribution offset is the
 * \mu, and scale is the \sigma in the gaussian distribution
 */
void cCtPDGenController::BuildStateOffsetScale(Eigen::VectorXd &out_offset,
                                               Eigen::VectorXd &out_scale) const
{
    cDeepMimicCharController::BuildStateOffsetScale(out_offset, out_scale);

    if (mEnablePhaseInput)
    {
        Eigen::VectorXd phase_offset;
        Eigen::VectorXd phase_scale;
        BuildStatePhaseOffsetScale(phase_offset, phase_scale);

        int phase_idx = GetStatePhaseOffset();
        int phase_size = GetStatePhaseSize();
        out_offset.segment(phase_idx, phase_size) = phase_offset;
        out_scale.segment(phase_idx, phase_size) = phase_scale;
    }
}

/**
 *
 */
void cCtPDGenController::BuildActionBounds(Eigen::VectorXd &out_min,
                                           Eigen::VectorXd &out_max) const
{
    int action_size = GetActionSize();
    out_min.resize(action_size);
    out_max.resize(action_size);

    int num_of_joints = mChar->GetNumJoints();
    int param_st = 0;
    for (int i = 0; i < num_of_joints; i++)
    {
        tVectorXd action_lb, action_ub;
        BuildJointActionBounds(i, action_lb, action_ub);
        int param_size = GetJointActionSize(i);
        out_min.segment(param_st, param_size) = action_lb;
        out_max.segment(param_st, param_size) = action_ub;
        param_st += param_size;
    }
    MIMIC_INFO("build action bound lb = {}, ub = {}", out_min.transpose(),
               out_max.transpose());
}

/**
 * \brief                   We assume that the correct policy action of the
 * character obey a gaussian distribution
 *
 *  So we have element wise value for action (offset and scale) = (mean and
 * variance)
 */
void cCtPDGenController::BuildActionOffsetScale(
    Eigen::VectorXd &out_offset, Eigen::VectorXd &out_scale) const
{
    int action_size = GetActionSize();
    out_offset.resize(action_size);
    out_scale.resize(action_size);

    int num_of_joints = mChar->GetNumJoints();
    int param_st = 0;
    for (int i = 0; i < num_of_joints; i++)
    {
        tVectorXd joint_offset, joint_scale;
        // BuildJointActionBounds(i, action_lb, action_ub);
        BuildJointActionOffsetScale(i, joint_offset, joint_scale);
        int param_size = GetJointActionSize(i);
        out_offset.segment(param_st, param_size) = joint_offset;
        out_scale.segment(param_st, param_size) = joint_scale;
        param_st += param_size;
    }
    MIMIC_INFO("build action offset = {}, scale = {}", out_offset.transpose(),
               out_scale.transpose());
}

const tVectorXd &cCtPDGenController::GetCurAction() { return mCurAction; }

bool cCtPDGenController::ParseParams(const Json::Value &json)
{
    bool succ = cCtController::ParseParams(json);
    SetupPDControllers(json, mGravity);
    return succ;
}

/**
 * \brief                   Calculate the control torque (can be generalized
 * force) which will be used directly in cSimCharBase::ApplyControlForces()
 * \param time_step         current timestep
 * \param out_tau           control force result
 */
void cCtPDGenController::UpdateBuildTau(double time_step,
                                        Eigen::VectorXd &out_tau)
{
    if (mEnableGuidedAction == true)
    {
        UpdateBuildTauGuided(time_step, out_tau);
    }
    else
    {
        UpdateBuildTauPD(time_step, out_tau);
    }

    PostUpdateBuildTau();
}

/**
 * \brief                   Update control force by PD control
 */
void cCtPDGenController::UpdateBuildTauPD(double dt, Eigen::VectorXd &out_tau)
{

    std::ofstream fout(pd_log, std::ios::app);
    tVectorXd target_q, target_qdot;
    fout << "---------------------\n";
    mPDGenController->GetPDTarget_q(target_q, target_qdot);
    fout << "target q = " << target_q.transpose() << std::endl;
    fout << "target qdot = " << target_qdot.transpose() << std::endl;

    // 1. Inverse dynamics to control the next pose
    /*
        M * (q_target - q_cur  - dt * qdot_cur) / (dt^2) - (Q_gravity - C * qdot
       )
    */
    // {
    //     auto gen_char = dynamic_cast<cSimCharacterGen *>(mChar);
    //     target_q.segment(0, 6) = gen_char->Getq().segment(0, 6);
    //     fout << "cur q = " << gen_char->Getq().transpose() << std::endl;
    //     fout << "cur qdot = " << gen_char->Getqdot().transpose() <<
    //     std::endl; tVectorXd residual_part =
    //         (target_q - gen_char->Getq() - dt * gen_char->Getqdot());
    //     tVectorXd Q_part = -gen_char->GetCoriolisMatrix() *
    //     gen_char->Getqdot(); Q_part[1] += mGravity[1] *
    //     gen_char->GetTotalMass(); out_tau =
    //         gen_char->GetMassMatrix() * residual_part / (dt * dt) - Q_part;
    //     out_tau.segment(0, 6).setZero();
    //     const double torque_lim = 20;
    //     fout << "raw tau = " << out_tau.transpose() << std::endl;
    //     out_tau = out_tau.cwiseMax(-torque_lim);
    //     out_tau = out_tau.cwiseMin(torque_lim);

    //     fout << "limited tau = " << out_tau.transpose() << std::endl;
    // }

    // 2. PD control
    {
        auto gen_char = static_cast<cSimCharacterGen *>(mChar);
        UpdatePDCtrls(dt, out_tau);
        fout << "cur q = " << gen_char->Getq().transpose() << std::endl;
        fout << "cur qdot = " << gen_char->Getqdot().transpose() << std::endl;
        fout << "ctrl tau = " << out_tau.transpose() << std::endl;
    }
}

/**
 * \brief                   Update control force by Action guiding
 */
void cCtPDGenController::UpdateBuildTauGuided(double time_step,
                                              Eigen::VectorXd &out_tau)
{
    // check initialize
    if (mInternalFrameId == 0)
    {
        std::cout << "load traj frame id " << mInternalFrameId << std::endl;
        mLoadInfo->LoadTrajV2(this->mChar, mGuidedTrajFile);
        // for (int i = 0; i < mLoadInfo->mTotalFrame; i++)
        // {
        //     std::cout << "frame " << i
        //               << " pose = " << mLoadInfo->mPoseMat.row(i) <<
        //               std::endl;
        // }
        // exit(1);
    }

    // 2. fetch & apply the control force
    if (mLoadInfo->mTotalFrame > mInternalFrameId)
    {
        out_tau = mLoadInfo->mActionMat.row(mInternalFrameId);
    }
    else
    {
        MIMIC_WARN("frame {} has exceed the guided range", mInternalFrameId);
    }

    // 3. compare the simulation result
    tVectorXd raw_pose = mLoadInfo->mCharPoseMat.row(mInternalFrameId),
              cur_pose = mChar->GetPose();
    tVectorXd diff = raw_pose - cur_pose;
    std::cout << "raw pose = " << raw_pose.transpose() << std::endl;
    std::cout << "cur pose = " << cur_pose.transpose() << std::endl;
    std::cout << "cur tau = " << out_tau.transpose() << std::endl;
    std::cout << "pose diff = " << diff.transpose() << std::endl;
    MIMIC_INFO("frame {} pose_diff norm {}, action norm {}", mInternalFrameId,
               diff.norm(), out_tau.norm());
}
void cCtPDGenController::PostUpdateBuildTau() { mInternalFrameId++; }
/**
 * \brief               Create the internal Stable PD controller by given
 * parameters
 */
#include "PDController.h"
#include "sim/SimItems/SimCharacterGen.h"
void cCtPDGenController::SetupPDControllers(const Json::Value &json,
                                            const tVector &gravity)
{
    Eigen::MatrixXd pd_params;
    bool succ = false;
    const std::string pd_key = "PDControllers";
    if (!json[pd_key].isNull())
    {
        succ = cPDController::LoadParams(json[pd_key], pd_params);
    }

    if (succ)
    {
        mPDGenController = new cImpPDGenController();
        // get kp & kd per joint
        tVectorXd kp_vec = pd_params.col(cPDController::eParam::eParamKp);
        tVectorXd kd_vec = pd_params.col(cPDController::eParam::eParamKd);
        MIMIC_INFO("kp vec = {}", kp_vec.transpose());
        MIMIC_INFO("kd vec = {}", kd_vec.transpose());
        mPDGenController->Init(dynamic_cast<cSimCharacterGen *>(mChar), kp_vec,
                               kd_vec);
        // exit(1);
    }

    mValid = succ;
    if (!mValid)
    {
        MIMIC_ERROR("Failed to initialize Ct-PDGen controller\n");
        mValid = false;
    }
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
    mPDGenController->UpdateControlForce(time_step, out_tau);
}

/**
 * \brief                   set the action(PD target) from outside
 * \param action
 */
void cCtPDGenController::ApplyAction(const Eigen::VectorXd &action)
{
    // 1. check the length of action (spherical - axis angle)
    MIMIC_ASSERT(GetActionSize() == action.size());

    // 2. convert action to PD target (pose)
    mCurAction.noalias() = action;
    mCurPDTargetPose.noalias() = mCurAction;
    ConvertActionToTargetPose(mCurPDTargetPose);

    // 3. convert pose to q, then set q and qdot to the stable PD controller
    SetPDTargets(mCurPDTargetPose);
}

/**
 * \brief                   build the action bounds (the action should not be
 * unlimited) for joints
 *
 */
void cCtPDGenController::BuildJointActionBounds(int joint_id,
                                                Eigen::VectorXd &out_min,
                                                Eigen::VectorXd &out_max) const
{
    const auto &joint = mChar->GetJoint(joint_id);
    switch (joint.GetType())
    {
    case cKinTree::eJointType::eJointTypeSpherical:
        BuildJointActionBoundsSpherical(joint_id, out_min, out_max);
        break;
    case cKinTree::eJointType::eJointTypeRevolute:
        BuildJointActionBoundsRevolute(joint_id, out_min, out_max);
        break;
    case cKinTree::eJointType::eJointTypeFixed:
        BuildJointActionBoundsFixed(joint_id, out_min, out_max);
        break;
    case cKinTree::eJointType::eJointTypeNone:
        BuildJointActionBoundsNone(joint_id, out_min, out_max);
        break;
    default:
        MIMIC_ERROR("Unsupported joint type {}", joint.GetType());
        break;
    }
}

/**
 * \brief                   the action of each joint obey a gaussian
 * distribution. The mean and variance of it is calculated here
 */
void cCtPDGenController::BuildJointActionOffsetScale(
    int joint_id, Eigen::VectorXd &out_offset, Eigen::VectorXd &out_scale) const
{
    const auto &joint = mChar->GetJoint(joint_id);
    switch (joint.GetType())
    {
    case cKinTree::eJointType::eJointTypeSpherical:
        BuildJointActionOffsetScaleSphereical(joint_id, out_offset, out_scale);
        break;
    case cKinTree::eJointType::eJointTypeNone:
        BuildJointActionOffsetScaleNone(joint_id, out_offset, out_scale);
        break;
    case cKinTree::eJointType::eJointTypeRevolute:
        BuildJointActionOffsetScaleRevolute(joint_id, out_offset, out_scale);
        break;
    case cKinTree::eJointType::eJointTypeFixed:
        BuildJointActionOffsetScaleFixed(joint_id, out_offset, out_scale);
        break;

    default:
        MIMIC_ERROR("Unsupported joint type {}", joint.GetType());
        break;
    }
}

cKinTree::eJointType cCtPDGenController::GetJointType(int joint_id) const
{
    return mChar->GetJoint(joint_id).GetType();
}

/**
 * \brief                   Convert the PD target pose to
 */
void cCtPDGenController::SetPDTargets(const Eigen::VectorXd &reduce_target_pose)
{
    MIMIC_ASSERT(GetActionSize() == reduce_target_pose.size());

    // expand to the full size
    const int root_pose_size = 7;
    tVectorXd full_target_pose =
        tVectorXd::Zero(root_pose_size + reduce_target_pose.size());
    full_target_pose.segment(root_pose_size, reduce_target_pose.size()) =
        reduce_target_pose;

    MIMIC_ASSERT(eSimCharacterType::Generalized == mChar->GetCharType());

    auto gen_char = static_cast<cSimCharacterGen *>(mChar);
    tVectorXd q_goal = gen_char->ConvertPoseToq(full_target_pose),
              qdot_goal = tVectorXd::Zero(gen_char->Getqdot().size());
    // std::cout << "target pose = " << full_target_pose.transpose() <<
    // std::endl; std::cout << "target q = " << q_goal.transpose() << std::endl;
    // exit(1);
    mPDGenController->SetPDTarget_q(q_goal, qdot_goal);
}

/**
 * \brief               For PD control strategy, the action is PD target
 * combined by quaternions
 */
int cCtPDGenController::GetActionSize() const
{
    int total_action_size = 0;
    for (int i = 0; i < mChar->GetNumJoints(); i++)
    {
        total_action_size += GetJointActionSize(i);
    }
    return total_action_size;
}

/**
 * \brief                   Get the size of action w.r.t current joint
 */
int cCtPDGenController::GetJointActionSize(int id) const
{
    MIMIC_ASSERT((id >= 0) && (id <= mChar->GetNumJoints() - 1))
    auto type = mChar->GetJoint(id).GetType();
    int size = 0;
    switch (type)
    {
    case cKinTree::eJointType::eJointTypeNone:
        size = 0;
        break;
    case cKinTree::eJointType::eJointTypeRevolute:
        size = 1;
        break;
    case cKinTree::eJointType::eJointTypeSpherical:
        size = 4;
        break;
    default:
        MIMIC_ERROR("unsupported type {}", type);
        break;
    }
    return size;
}

/**
 *\brief                    Build joint action bounds for revolute joints
 * mean = (ub + lb) / 2,
 * range = (ub - lb)
 * revolute action_lb = mean - 2 * range
 * revolute action_ub = mean + 2 * range
 *
 */
void cCtPDGenController::BuildJointActionBoundsRevolute(
    int joint_id, Eigen::VectorXd &out_min, Eigen::VectorXd &out_max) const
{
    MIMIC_ASSERT(mChar->GetJoint(joint_id).GetType() ==
                 cKinTree::eJointType::eJointTypeRevolute);

    const auto &joint = mChar->GetJoint(joint_id);
    int size = GetJointActionSize(joint_id);
    MIMIC_ASSERT(size == 1);
    out_min = tVectorXd::Zero(size);
    out_max = tVectorXd::Zero(size);
    const tVectorXd &lim_low = joint.GetLimLow(),
                    &lim_high = joint.GetLimHigh();
    tVectorXd mean = (lim_high + lim_low) / 2;
    tVectorXd range = (lim_high - lim_low);
    out_min[0] = (mean - 2 * range)[0];
    out_max[0] = (mean + 2 * range)[0];
    MIMIC_ASSERT(out_min.size() == size);
    MIMIC_ASSERT(out_max.size() == size);
}

/**
 * \brief                   Build joint action bound for spherical joints
 */
void cCtPDGenController::BuildJointActionBoundsSpherical(
    int joint_id, Eigen::VectorXd &out_min, Eigen::VectorXd &out_max) const
{
    MIMIC_ASSERT(mChar->GetJoint(joint_id).GetType() ==
                 cKinTree::eJointType::eJointTypeSpherical);

    const auto &joint = mChar->GetJoint(joint_id);
    int size = GetJointActionSize(joint_id);
    MIMIC_ASSERT(size == 4);
    out_min = -1 * tVectorXd::Ones(size);
    out_max = 1 * tVectorXd::Ones(size);

    {
        double lim_low = joint.GetLimLow().minCoeff(),
               lim_high = joint.GetLimHigh().maxCoeff();
        double mean = (lim_high + lim_low) / 2;
        double range = (lim_high - lim_low);
        out_min[0] = (mean - 2 * range);
        out_max[0] = (mean + 2 * range);
    }
    MIMIC_INFO("spherical joint {} action lb = {}, ub = {}", joint_id,
               out_min.transpose(), out_max.transpose());

    MIMIC_ASSERT(out_min.size() == size);
    MIMIC_ASSERT(out_max.size() == size);
}

/**
 * \brief                   None joint and fixed joint have no action
 */
void cCtPDGenController::BuildJointActionBoundsNone(
    int joint_id, Eigen::VectorXd &out_min, Eigen::VectorXd &out_max) const
{
    MIMIC_ASSERT(mChar->GetJoint(joint_id).GetType() ==
                 cKinTree::eJointType::eJointTypeNone);
    int size = GetJointActionSize(joint_id);
    MIMIC_ASSERT(size == 0);
    out_min.resize(size);
    out_max.resize(size);
}
void cCtPDGenController::BuildJointActionBoundsFixed(
    int joint_id, Eigen::VectorXd &out_min, Eigen::VectorXd &out_max) const
{
    MIMIC_ASSERT(mChar->GetJoint(joint_id).GetType() ==
                 cKinTree::eJointType::eJointTypeFixed);
    int size = GetJointActionSize(joint_id);
    MIMIC_ASSERT(size == 0);
    out_min.resize(size);
    out_max.resize(size);
}

/**
 * \brief               Spherical joints, offset = 0, scale = 1 beside the first
 * and the last number
 */
void cCtPDGenController::BuildJointActionOffsetScaleSphereical(
    int joint_id, Eigen::VectorXd &out_offset, Eigen::VectorXd &out_scale) const
{
    MIMIC_ASSERT(mChar->GetJoint(joint_id).GetType() ==
                 cKinTree::eJointType::eJointTypeSpherical);
    int action_size = GetJointActionSize(joint_id);
    MIMIC_ASSERT(action_size == 4);
    out_offset = tVectorXd::Zero(action_size);

    out_scale = tVectorXd::Ones(action_size);

    // set speical value for the [0] and [3] numer
    {
        const auto &joint = mChar->GetJoint(joint_id);
        tVectorXd lim_low = joint.GetLimLow(), lim_high = joint.GetLimHigh();
        out_scale[0] = 1.0 / (2 * (lim_high.maxCoeff() - lim_low.minCoeff()));
        out_offset[3] = -0.2;
    }
    MIMIC_INFO("spherical joint {} offset {}, sclae {}", joint_id,
               out_offset.transpose(), out_scale.transpose());
}
void cCtPDGenController::BuildJointActionOffsetScaleRevolute(
    int joint_id, Eigen::VectorXd &out_offset, Eigen::VectorXd &out_scale) const
{
    MIMIC_ASSERT(mChar->GetJoint(joint_id).GetType() ==
                 cKinTree::eJointType::eJointTypeRevolute);
    int action_size = GetJointActionSize(joint_id);
    MIMIC_ASSERT(action_size == 1);
    out_offset = tVectorXd::Zero(action_size);
    out_scale = tVectorXd::Ones(action_size);

    const auto &joint = mChar->GetJoint(joint_id);
    tVectorXd lim_low = joint.GetLimLow(), lim_high = joint.GetLimHigh();
    out_offset[0] = -0.5 * (lim_high[0] + lim_low[0]);
    out_scale[0] = 1 / (2 * (lim_high[0] - lim_low[0]));
    MIMIC_INFO("revolute joint {} offset {} scale {}", joint_id,
               out_offset.transpose(), out_scale.transpose());
}
void cCtPDGenController::BuildJointActionOffsetScaleFixed(
    int joint_id, Eigen::VectorXd &out_offset, Eigen::VectorXd &out_scale) const
{
    MIMIC_ASSERT(mChar->GetJoint(joint_id).GetType() ==
                 cKinTree::eJointType::eJointTypeFixed);
    int action_size = GetJointActionSize(joint_id);
    MIMIC_ASSERT(action_size == 0);
    out_offset.resize(action_size);
    out_scale.resize(action_size);
}
void cCtPDGenController::BuildJointActionOffsetScaleNone(
    int joint_id, Eigen::VectorXd &out_offset, Eigen::VectorXd &out_scale) const
{
    MIMIC_ASSERT(mChar->GetJoint(joint_id).GetType() ==
                 cKinTree::eJointType::eJointTypeNone);
    int action_size = GetJointActionSize(joint_id);
    MIMIC_ASSERT(action_size == 0);
    out_offset.resize(action_size);
    out_scale.resize(action_size);
}

/**
 * \brief               Convert action (axis angle in spherical joints, [theta,
 * ax, ay, az]) to PD Target pose(quaternion in spherical joints)
 */
void cCtPDGenController::ConvertActionToTargetPose(tVectorXd &out_theta) const
{
    MIMIC_ASSERT(out_theta.size() == GetActionSize());
    int st_pos = 0;
    for (int i = 0; i < mChar->GetNumJoints(); i++)
    {
        int param_size = GetJointActionSize(i);
        const auto &joint = mChar->GetJoint(i);

        // convert axis angle to quaternion for spherical joints
        if (cKinTree::eJointType::eJointTypeSpherical == joint.GetType())
        {
            MIMIC_ASSERT(param_size == 4);
            tVector axis_angle =
                out_theta.segment(st_pos, param_size).segment(0, 4);
            double theta = axis_angle[0];
            tVector axis = cMathUtil::Expand(axis_angle.segment(1, 3), 0);

            out_theta.segment(st_pos, param_size) = cMathUtil::QuatToVec(
                cMathUtil::AxisAngleToQuaternion(axis, theta));
            // MIMIC_DEBUG("convert axis angle {} to quaternion {}",
            //             axis_angle.transpose(),
            //             out_theta.segment(st_pos, param_size).transpose());
        }

        st_pos += param_size;
    }
}

void cCtPDGenController::ConvertTargetPoseToActionFullsize(tVectorXd &pd_target)
{

    // std::cout << "pd target size " << pd_target.size() << " act size "
    //           << GetActionSize() << std::endl;
    MIMIC_ASSERT(pd_target.size() == this->GetActionSize());
    int num_of_joints = mChar->GetNumJoints();
    int st_pos = 0;

    // from the first joint
    for (int i = 1; i < num_of_joints; i++)
    {
        int size = GetJointActionSize(i);
        tVectorXd target_pose = pd_target.segment(st_pos, size);
        // MIMIC_INFO("joint {}  target pose {}", i, target_pose.transpose());
        ConvertTargetPoseToAction(i, target_pose);
        pd_target.segment(st_pos, size) = target_pose;
        // MIMIC_INFO("joint {}  action {}", i, target_pose.transpose());
        st_pos += size;
    }
}

void cCtPDGenController::ConvertTargetPoseToAction(
    int joint_id, Eigen::VectorXd &out_theta) const
{
#if defined(ENABLE_PD_SPHERE_AXIS)
    cKinTree::eJointType joint_type = GetJointType(joint_id);
    if (joint_type == cKinTree::eJointTypeSpherical)
    {
        // raw input quaternion = [x, y, z, w]
        // tQuaternion quater = tQuaternion(out_theta[3], out_theta[0],
        // out_theta[1], out_theta[2]);

        // 2020/05/12 revised by Xudong: now input quaternion should be [w,
        // x, y, z]
        tQuaternion quater = cMathUtil::VecToQuat(out_theta);

        quater.normalize();
        tVector axis_angle =
            cMathUtil::QuaternionToAxisAngle(quater); //[theta, ax, ay, az]
        out_theta[0] = axis_angle.norm();
        axis_angle.normalize();
        out_theta[1] = axis_angle[0];
        out_theta[2] = axis_angle[1];
        out_theta[3] = axis_angle[2];
    }
#endif
}
