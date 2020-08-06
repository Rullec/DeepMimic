#include "SimBodyJointGen.h"
#include "BulletGenDynamics/btGenModel/Joint.h"
#include "SimCharacterGen.h"
#include "util/LogUtil.hpp"

cSimBodyJointGen::cSimBodyJointGen() {}
cSimBodyJointGen::~cSimBodyJointGen() { Clear(); }

void cSimBodyJointGen::Init(std::shared_ptr<cWorldBase> &world,
                            cRobotModelDynamics *multibody, int joint_id)
{
    mWorld = world;
    mRobotModel = multibody;

    mJointId = joint_id;
    mJoint = static_cast<Joint *>(multibody->GetJointById(joint_id));
    mJointType = FetchJointType();

    InitLimit();
}

void cSimBodyJointGen::Clear() { ClearTau(); }
bool cSimBodyJointGen::IsValid() const { return true; }

cKinTree::eJointType cSimBodyJointGen::GetType() const { return mJointType; }

void cSimBodyJointGen::CalcWorldRotation(tVector &out_axis,
                                         double &out_theta) const
{
    cMathUtil::RotMatToAxisAngle(
        cMathUtil::ExpandMat(mJoint->GetWorldOrientation()), out_axis,
        out_theta);
}
tQuaternion cSimBodyJointGen::CalcWorldRotation() const
{

    return cMathUtil::RotMatToQuaternion(
        cMathUtil::ExpandMat(mJoint->GetWorldOrientation()));
}

tMatrix cSimBodyJointGen::BuildWorldTrans() const
{
    return mJoint->GetGlobalTransform();
}
bool cSimBodyJointGen::IsRoot() const
{
    return mJointType == cKinTree::eJointType::eJointTypeNone;
}

void cSimBodyJointGen::AddTau(const Eigen::VectorXd &tau)
{
    cSimBodyJoint::AddTau(tau);
}
const cSpAlg::tSpVec &cSimBodyJointGen::GetTau() const { return mTotalTau; }
void cSimBodyJointGen::ApplyTau() { cSimBodyJoint::ApplyTau(); }
void cSimBodyJointGen::ClearTau() { mTotalTau.setZero(); }

tVector cSimBodyJointGen::CalcWorldPos() const
{
    return cMathUtil::Expand(mJoint->GetWorldPos(), 0);
}
tVector cSimBodyJointGen::CalcWorldPos(const tVector &local_pos) const
{
    MIMIC_ASSERT(std::fabs(local_pos[3] - 1.0) < 1e-10);
    return mJoint->GetGlobalTransform() * local_pos;
}
tVector cSimBodyJointGen::CalcWorldVel() const
{
    return cMathUtil::Expand(mJoint->GetJointVel(), 0);
}
tVector cSimBodyJointGen::CalcWorldVel(const tVector &local_pos) const
{
    tMatrixXd jac;
    mJoint->ComputeJacobiByGivenPoint(jac, local_pos);
    return cMathUtil::Expand(jac * mRobotModel->Getqdot(), 0);
}
tVector cSimBodyJointGen::CalcWorldAngVel() const
{
    return cMathUtil::Expand(mJoint->GetJointOmega(), 0);
}

double cSimBodyJointGen::GetTorqueLimit() const { return mTorqueLimit; }
double cSimBodyJointGen::GetForceLimit() const { return mForceLimit; }
void cSimBodyJointGen::SetTorqueLimit(double lim)
{
    mTorqueLimit = lim;
    MIMIC_DEBUG("set joint torque limit {} for joint {}", mTorqueLimit,
                mJointId);
}
void cSimBodyJointGen::SetForceLimit(double lim) { mForceLimit = lim; }

/**
 * \brief               Get the joint position in parent link frame
 */
tVector cSimBodyJointGen::GetParentPos() const
{
    MIMIC_ERROR("this method should not be called");
    return tVector::Zero();
}

/**
 * \brief               Get the joint position in child link frame
 */
tVector cSimBodyJointGen::GetChildPos() const
{
    MIMIC_ERROR("this method should not be called");
    return tVector::Zero();
}

tQuaternion cSimBodyJointGen::GetChildRot() const
{
    MIMIC_ERROR("this method should not be called");
    return tQuaternion::Identity();
}

tMatrix cSimBodyJointGen::BuildJointChildTrans() const
{
    MIMIC_ERROR("this method should not be called");
    return tMatrix::Identity();
}

tMatrix cSimBodyJointGen::BuildJointParentTrans() const
{
    MIMIC_ERROR("this method should not be called");
    return tMatrix::Identity();
}

/**
 * \brief           Get the axis in world frame
 *
 *      1. the "axis" of joints are assumed to be (1, 0, 0) in local frame
 *              (implemented in GetAxisRel)
 *
 */
tVector cSimBodyJointGen::CalcAxisWorld() const
{
    tVector axis_rel = GetAxisRel();
    axis_rel[3] = 0;
    tMatrix trans = BuildWorldTrans();
    tVector axis = trans * axis_rel;
    return axis;
}
tVector cSimBodyJointGen::GetAxisRel() const { return tVector(1, 0, 0, 0); }

int cSimBodyJointGen::GetPoseSize() const
{
    int size = -1;
    switch (GetType())
    {
    case cKinTree::eJointType::eJointTypeFixed:
        size = 0;
        break;
    case cKinTree::eJointType::eJointTypeSpherical:
        size = 4;
        break;
    case cKinTree::eJointType::eJointTypeNone:
        size = 7;
        break;
    case cKinTree::eJointType::eJointTypeRevolute:
        size = 1;
        break;

    default:
        MIMIC_ERROR("Unsupported joint type {}", GetType());
        break;
    }
    return size;
}

bool cSimBodyJointGen::HasParent() const { return IsRoot() == true; }
bool cSimBodyJointGen::HasChild() const { return true; }

const std::shared_ptr<cSimBodyLink> &cSimBodyJointGen::GetParent() const
{
    return dynamic_cast<cSimCharacterGen *>(mRobotModel)
        ->GetBodyPart(mJoint->GetParentId());
}
const std::shared_ptr<cSimBodyLink> &cSimBodyJointGen::GetChild() const
{
    // child link id is the same is its joint id
    return dynamic_cast<cSimCharacterGen *>(mRobotModel)->GetBodyPart(mJointId);
}

/**
 * \brief               Given the max torque, clamp the input
 */
void cSimBodyJointGen::ClampTotalTorque(tVector &out_torque) const
{
    double mag = out_torque.norm();
    if (mag > mTorqueLimit)
    {
        out_torque *= mTorqueLimit / mag;
        MIMIC_WARN("joint {} torque lim {}, cur torque = {}", mJointId,
                   mTorqueLimit, mag);
    }
}
void cSimBodyJointGen::ClampTotalForce(tVector &out_force) const
{
    double mag = out_force.norm();
    if (mag > mForceLimit)
    {
        out_force *= mForceLimit / mag;
        MIMIC_WARN("joint {} force lim {}, cur force = {}", mJointId,
                   mForceLimit, mag);
    }
}

const tVector &cSimBodyJointGen::GetLimLow() const { return mLimLow; }
const tVector &cSimBodyJointGen::GetLimHigh() const { return mLimHigh; }
bool cSimBodyJointGen::HasJointLim() const
{
    for (int i = 0; i < 3; i++)
    {
        if (mLimHigh[i] < mLimLow[i])
            return false;
    }
    return true;
}

/**
 * \brief               Get the parameter size of this joint
 *
 * In raw DeepMimic, the parameter size is typically used to determine not only
 * the number of control parameter, but also the number of pose it occupied.
 *
 * But now, after using the generalized coordinate framework, the param size is
 * changed to the number of DOF in this joint it can still be used to determine
 * the controler parameter, but not equal to the numer of pose it occupied
 * anymore.
 */
int cSimBodyJointGen::GetParamSize() const { return mJoint->GetNumOfFreedom(); }
void cSimBodyJointGen::BuildPose(Eigen::VectorXd &out_pose) const
{
    MIMIC_ERROR("shouldn't called\n");
}
void cSimBodyJointGen::BuildVel(Eigen::VectorXd &out_vel) const
{
    MIMIC_ERROR("shouldn't called\n");
}
void cSimBodyJointGen::SetPose(const Eigen::VectorXd &pose)
{
    MIMIC_ERROR("shouldn't called\n");
}
void cSimBodyJointGen::SetVel(const Eigen::VectorXd &vel)
{
    MIMIC_ERROR("shouldn't called\n");
}

tVector cSimBodyJointGen::GetTotalTorque() const
{
    return cSpAlg::GetOmega(mTotalTau);
}
tVector cSimBodyJointGen::GetTotalForce() const
{
    return cSpAlg::GetV(mTotalTau);
}

cKinTree::eJointType cSimBodyJointGen::FetchJointType() const
{
    cKinTree::eJointType type = cKinTree::eJointType::eJointTypeFixed;
    switch (mJoint->GetJointType())
    {
    case JointType::REVOLUTE_JOINT:
        type = cKinTree::eJointType::eJointTypeRevolute;
        break;
    case JointType::NONE_JOINT:
        type = cKinTree::eJointType::eJointTypeNone;
        break;
    case JointType::FIXED_JOINT:
        type = cKinTree::eJointType::eJointTypeFixed;
        break;
    case JointType::SPHERICAL_JOINT:
        type = cKinTree::eJointType::eJointTypeSpherical;
        break;
    default:
        MIMIC_ERROR("invalid jont type {}", mJoint->GetJointType());
        break;
    }
    return type;
}

tVector cSimBodyJointGen::CalcParentLocalPos(const tVector &local_pos) const
{
    MIMIC_ERROR("shouldn't be called");
    return tVector::Zero();
}
tVector cSimBodyJointGen::CalcChildLocalPos(const tVector &local_pos) const
{

    MIMIC_ERROR("shouldn't be called");
    return tVector::Zero();
}

void cSimBodyJointGen::SetTotalTorque(const tVector &torque)
{
    cSimBodyJoint::SetTotalTorque(torque);
}
void cSimBodyJointGen::SetTotalForce(const tVector &force)
{
    cSimBodyJoint::SetTotalForce(force);
}

/**
 * \brief                   Apply current torque to revolute joint
 *      1. clamp & set the torque again
 *      2. add this torque into the RobotModel
 */
void cSimBodyJointGen::ApplyTauRevolute()
{
    tVector torque = GetTotalTorque();
    ClampTotalForce(torque);
    SetTotalTorque(torque);

    double world_scale = mWorld->GetScale();

    mRobotModel->ApplyJointTorque(mJointId, world_scale * world_scale * torque);
}
void cSimBodyJointGen::ApplyTauPlanar()
{
    MIMIC_ERROR("hasn't been implemented\n");
}
void cSimBodyJointGen::ApplyTauPrismatic()
{
    MIMIC_ERROR("hasn't been implemented\n");
}
void cSimBodyJointGen::ApplyTauSpherical()
{
    tVector torque = GetTotalTorque();
    ClampTotalForce(torque);
    SetTotalTorque(torque);

    double world_scale = mWorld->GetScale();

    mRobotModel->ApplyJointTorque(mJointId, world_scale * world_scale * torque);
}

Joint *cSimBodyJointGen::GetInternalJoint() const { return mJoint; }

// static std::string toString(const Eigen::MatrixXd &mat)
// {
//     std::stringstream ss;
//     ss << mat;
//     return ss.str();
// }

/**
 * \brief           Init force limit, joint limit, joint angle limit (low and
 * high)
 */
void cSimBodyJointGen::InitLimit()
{
    mForceLimit = std::numeric_limits<double>::infinity();
    mTorqueLimit = mJoint->GetTorqueLim();
    mLimLow.setZero();
    mLimHigh.setZero();

    // for root joint, limit is not possible
    if (IsRoot() == true)
        return;

    for (int i = 0; i < mJoint->GetNumOfFreedom(); i++)
    {
        mLimLow[i] = mJoint->GetFreedoms(i)->lb;
        mLimHigh[i] = mJoint->GetFreedoms(i)->ub;
    }

    MIMIC_DEBUG("for joint {}, force limit {}, torque lim {}, freedom lowlim "
                "{}, freedom highlim {}",
                mJointId, mForceLimit, mTorqueLimit, mLimLow.transpose(),
                mLimHigh.transpose());
}

double cSimBodyJointGen::GetJointDiffWeight() const
{
    return mJoint->GetDiffWeight();
}