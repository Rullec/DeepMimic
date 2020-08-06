#include "SimCharacterGen.h"
#include "BulletGenDynamics/btGenModel/Joint.h"
#include "BulletGenDynamics/btGenModel/Link.h"
#include "BulletGenDynamics/btGenModel/RobotModelDynamics.h"
#include "BulletGenDynamics/btGenModel/RootJoint.h"
#include "SimBodyJointGen.h"
#include "SimBodyLinkGen.h"
#include "util/LogUtil.hpp"
#include <iostream>

tVector ConvertAxisAngleVelToEulerAngleVel(const tVector &aa_vel)
{
    MIMIC_ASSERT(std::fabs(aa_vel[3]) < 1e-10);
    double dt = 1e-4;
    return cMathUtil::AxisAngleToEuler(aa_vel.normalized(),
                                       aa_vel.norm() * dt) /
           dt;
}

tVector ConvertEulerAngleVelToAxisAngleVel(const tVector &ea_vel)
{
    MIMIC_ASSERT(std::fabs(ea_vel[3]) < 1e-10);
    // std::cout << "euler anble vel = " << ea_vel.transpose() << std::endl;
    double dt = 1e-4;

    tVector aa_vel =
        cMathUtil::EulerangleToAxisAngle(ea_vel * dt, eRotationOrder::XYZ) / dt;
    // std::cout << "aa vel = " << aa_vel.transpose() << std::endl;
    return aa_vel;
}

cSimCharacterGen::cSimCharacterGen()
{
    mController = nullptr;
    mLinkGenArray.clear();
    mJointGenArray.clear();
    // mLinkBaseArray.clear();
    // mJointBaseArray.clear();
}

bool cSimCharacterGen::Init(const std::shared_ptr<cWorldBase> &world,
                            const cSimCharacterBase::tParams &params)
{
    MIMIC_DEBUG("init generalized char, world sclae is {}", world->GetScale());
    mWorld = world;
    cRobotModelDynamics::Init(params.mCharFile.c_str(), world->GetScale(),
                              ModelType::JSON);
    btDiscreteDynamicsWorld *dis_world =
        dynamic_cast<btDiscreteDynamicsWorld *>(
            world->GetInternalWorld().get());
    MIMIC_ASSERT(nullptr != dis_world);
    cRobotModelDynamics::InitSimVars(dis_world, true, true);

    // std::cout << "q size = " << mq.size() << std::endl;
    // std::cout << "qdot size = " << mqdot.size() << std::endl;
    // First build links, then build joints
    BuildBodyLinks();
    BuildJoints();

    InitDefaultState();
    LoadDrawShapeDefs(params.mCharFile, mDrawShapeDefs);
    // test SetRootTransform method
    // Test();
    // exit(0);
    return true;
}

/**
 * \brief           Clear this character
 *
 *      This function will delete this character from this world
 *      and clear the controller
 */
void cSimCharacterGen::Clear()
{
    cCharacter::Clear();

    RemoveFromWorld();

    if (HasController())
    {
        mController->Clear();
    }
}

/**
 * \brief           Reset
 *      This function will reset the controller, and clear all joint torques
 */
void cSimCharacterGen::Reset()
{
    cCharacter::Reset();
    if (HasController())
    {
        mController->Reset();
    }
    ClearJointTorques();
}

/**
 * \brief           Update this character
 *
 *      1. clear joint torques
 *      2. update controller
 *      3. update joints' info
 */
void cSimCharacterGen::Update(double time_step)
{
    ClearJointTorques();

    if (HasController())
        mController->Update(time_step);

    // dont clear torques until next frame since they can be useful for
    // visualization
    UpdateJoints();
}

/**
 * \brief           Update some simulation variables after being updated
 *
 *  1. update the link position/velocity
 *  2. update the char pose/ pose_vel
 *  3. post update the controller
 */
void cSimCharacterGen::PostUpdate(double time_step)
{
    BuildPose(mPose);
    BuildVel(mVel);
    if (HasController())
        mController->PostUpdate(time_step);
}

/**
 * \brief               Get the number of freedoms
 *
 * Note that this function can be used when initializing the action size
 */
int cSimCharacterGen::GetNumDof() const { return GetNumOfFreedom(); }

/**
 * \brief               Get the "parameter size" w.r.t a joint
 *
 * return the number of paramerters in order to fully determine a joint.
 * This function is used to determine the action size.
 *
 * parameter size = joint dof
 * for root joint = 6
 * for spherical joint = 3
 */
int cSimCharacterGen::GetParamSize(int joint_id) const
{
    MIMIC_ASSERT(joint_id < mJointGenArray.size() && joint_id >= 0);
    return mJointGenArray[joint_id]->GetParamSize();
}

/**
 *
 */
int cSimCharacterGen::GetParamOffset(int joint_id) const
{
    MIMIC_WARN("need to be tested");
    return mJointGenArray[joint_id]
        ->GetInternalJoint()
        ->GetPrevFreedomIds()
        .size();
}

int cSimCharacterGen::GetRootID() const { return 0; }

/**
 * \brief           Get the position of root link in world frame
 */
tVector cSimCharacterGen::GetRootPos() const
{
    int root_id = 0;
    auto root_joint = mJointGenArray[root_id];
    MIMIC_ASSERT(cKinTree::eJointType::eJointTypeNone == root_joint->GetType());

    return mLinkGenArray[root_id]->GetPos();
}

/**
 * \brief           Get the rotation of root joint, axis angle
 *
 */
void cSimCharacterGen::GetRootRotation(tVector &out_axis,
                                       double &out_theta) const
{
    tQuaternion rot = GetRootRotation();
    cMathUtil::QuaternionToAxisAngle(rot, out_axis, out_theta);
}

/**
 * \brief           Get the rotation of root joint, quaternion
 *  Note that, this function are trying to get the pure rotaion of root joint.
 * It is different from the world rotaion of root joint when it has a rotation
 * in rest pose
 */
tQuaternion cSimCharacterGen::GetRootRotation() const
{
    int root_id = 0;
    auto root_joint = mJointGenArray[root_id];
    return root_joint->CalcWorldRotation();
}

/**
 * \brief           Get the velocity of root link in world frame
 *
 *      It is simply the first 3 elements of generalized coordinate
 */
tVector cSimCharacterGen::GetRootVel() const
{
    return cMathUtil::Expand(mqdot.segment(0, 3), 0);
}

/**
 * \brief           Get the angular vel of root link in world frame
 *
 *      w = Jw * qdot
 */
tVector cSimCharacterGen::GetRootAngVel() const
{
    tMatrixXd root_jac;
    int root_id = 0;
    ComputeJacobiByGivenPointTotalDOFWorldFrame(
        root_id, GetRootPos().segment(0, 3), root_jac);
    return cMathUtil::Expand((root_jac * mqdot), 3);
}

/**
 *  Should not be called for Generalized character
 */
const Eigen::MatrixXd &cSimCharacterGen::GetBodyDefs() const
{
    MIMIC_ASSERT_INFO(false, "this function should not be called\n");
    return Eigen::MatrixXd::Zero(0, 0);
}

/**
 * \brief           set the root position of character
 *
 * Only update the transform of this chain
 */
void cSimCharacterGen::SetRootPos(const tVector &pos)
{
    mq.segment(0, 3) = pos.segment(0, 3);
    Apply(mq, false);
}

/**
 * \brief           set the root rotation, axis angle to euler angle
 */
void cSimCharacterGen::SetRootRotation(const tVector &axis, double theta)
{
    tQuaternion q = cMathUtil::AxisAngleToQuaternion(axis, theta);
    SetRootTransform(GetRootPos(), q);
}

void cSimCharacterGen::SetRootRotation(const tQuaternion &q_)
{
    SetRootTransform(GetRootPos(), q_);
}

void cSimCharacterGen::SetRootTransform(const tVector &pos,
                                        const tQuaternion &rot)
{
    // MIMIC_ERROR("this functions needs to be tested well.\n");

    // 1. record old info
    int num_of_links = GetNumOfLinks();
    auto link = static_cast<Link *>(mLinkGenArray[0]->GetInternalLink());
    tMatrix old_rotation = tMatrix::Zero();
    old_rotation.block(0, 0, 3, 3) = link->GetWorldOrientation();
    tVector old_link_vel = cMathUtil::Expand(link->GetLinkVel(), 0),
            old_link_omega = cMathUtil::Expand(link->GetLinkOmega(), 0);

    // 2. set up new rotation, calculate new vel and new omega
    tMatrix new_orientation = cMathUtil::RotMat(rot);
    tMatrix diff_rot = new_orientation * old_rotation.transpose();
    tVector diff_rot_euler_angles =
        cMathUtil::RotMatToEuler(diff_rot, eRotationOrder::XYZ);

    tMatrix new_rotation =
        new_orientation *
        cMathUtil::ExpandMat(link->GetMeshRotation()).transpose();
    // std::cout << "new rot = \n" << new_rotation << std::endl;

    tVector new_euler_angles = cMathUtil::QuaternionToEulerAngles(
        cMathUtil::RotMatToQuaternion(new_rotation), eRotationOrder::XYZ);
    tVector new_link_vel =
        diff_rot * old_link_vel; // diff rotation in world frame
    tVector new_link_omega = diff_rot * old_link_omega;

    std::vector<tVector3d> old_link_vel_array(num_of_links, tVector3d::Zero()),
        old_link_omega_array(num_of_links, tVector3d::Zero());
    std::vector<tMatrix3d> old_rotation_array(num_of_links, tMatrix3d::Zero());
    for (int i = 0; i < num_of_links; i++)
    {
        auto link = static_cast<Link *>(mLinkGenArray[i]->GetInternalLink());
        old_rotation_array[i] = link->GetWorldOrientation();
        old_link_omega_array[i] = link->GetLinkOmega();
        old_link_vel_array[i] = link->GetLinkVel();
    }

    // 3. given new q into the result and update
    mq.segment(0, 3) = pos.segment(0, 3);
    mq.segment(3, 3) = new_euler_angles.segment(0, 3);
    Apply(mq, true);

    // 4. given new link vel to qdot
    tVectorXd qdot = Getqdot();
    // std::cout << "root link jkv = \n" << link->GetJKv() << std::endl;
    qdot.segment(3, 3) = link->GetJKw().block(0, 3, 3, 3).inverse() *
                         new_link_omega.segment(0, 3);
    qdot.segment(0, 3) = new_link_vel.segment(0, 3) -
                         link->GetJKv().block(0, 3, 3, 3) * qdot.segment(3, 3);
    Setqdot(qdot);
}

void cSimCharacterGen::SetRootVel(const tVector &vel)
{
    mqdot.segment(0, 3) = vel.segment(0, 3);
    Setqdot(mqdot);
}

void cSimCharacterGen::SetRootAngVel(const tVector &ang_vel)
{
    mqdot.segment(3, 3) = mLinkGenArray[0]
                              ->GetInternalLink()
                              ->GetJKw()
                              .block(0, 3, 3, 3)
                              .inverse() *
                          ang_vel.segment(0, 3);
    Setqdot(mqdot);
}

tQuaternion cSimCharacterGen::CalcHeadingRot() const
{
    tVector ref_dir = tVector(1, 0, 0, 0);
    tQuaternion root_rot = GetRootRotation();
    tVector rot_dir = cMathUtil::QuatRotVec(root_rot, ref_dir);
    double heading = std::atan2(-rot_dir[2], rot_dir[0]);
    return cMathUtil::AxisAngleToQuaternion(tVector(0, 1, 0, 0), heading);
}

int cSimCharacterGen::GetNumBodyParts() const { return GetNumOfLinks(); }

/**
 * \brief               Set Character pose given DeepMimic pose
 * \param pose          the pose vector given in DeepMimic kinchar format
 *      1. convert DeepMimic pose into generalized pose
 */
void cSimCharacterGen::SetPose(const Eigen::VectorXd &pose)
{
    tVectorXd q = ConvertPoseToq(pose);
    SetqAndqdot(q, mqdot);
}
void cSimCharacterGen::SetVel(const Eigen::VectorXd &vel)
{
    tVectorXd qdot = ConvertPosevelToqdot(vel);
    Setqdot(qdot);
}

tVector cSimCharacterGen::CalcJointPos(int joint_id) const
{
    auto joint = mJointGenArray[joint_id];
    return joint->CalcWorldPos();
}

/**
 * \brief               Get the velocity of literally joint COM
 * \param joint_id
 */
tVector cSimCharacterGen::CalcJointVel(int joint_id) const
{
    return cMathUtil::Expand(mJointGenArray[joint_id]->CalcWorldVel(), 0);
}
void cSimCharacterGen::CalcJointWorldRotation(int joint_id, tVector &out_axis,
                                              double &out_theta) const
{
    mJointGenArray[joint_id]->CalcWorldRotation(out_axis, out_theta);
}
tQuaternion cSimCharacterGen::CalcJointWorldRotation(int joint_id) const
{
    return mJointGenArray[joint_id]->CalcWorldRotation();
}
tMatrix cSimCharacterGen::BuildJointWorldTrans(int joint_id) const
{
    return mJointGenArray[joint_id]->BuildWorldTrans();
}

tVector cSimCharacterGen::CalcCOM() const
{
    return cMathUtil::Expand(GetCoMPosition(), 1);
}

tVector cSimCharacterGen::CalcCOMVel() const
{
    tVector COM_vel = tVector::Zero();
    for (int i = 0; i < GetNumOfLinks(); i++)
    {
        auto link = mLinkGenArray[i];
        COM_vel += link->GetLinearVelocity() * link->GetMass();
    }

    return COM_vel;
}

/**
 * \brief           Calculate the Axis Aligned Bounding Box(AABB) for the whold
 * character
 *
 */
void cSimCharacterGen::CalcAABB(tVector &out_min, tVector &out_max) const
{
    out_max.fill(-INF);
    out_min.fill(INF);

    for (int i = 0; i < GetNumOfLinks(); i++)
    {
    }
}

tVector cSimCharacterGen::GetSize() const { return tVector::Zero(); }

const cSimBodyJoint &cSimCharacterGen::GetJoint(int joint_id) const
{
    MIMIC_ERROR("no");
    return *(new cSimBodyJoint());
}
cSimBodyJoint &cSimCharacterGen::GetJoint(int joint_id)
{
    MIMIC_ERROR("no");
    return *(new cSimBodyJoint());
}
void cSimCharacterGen::GetChildJoint(int joint_id,
                                     Eigen::VectorXd &out_child_id)
{
}

int cSimCharacterGen::GetParentJoint(int joint_id)
{
    MIMIC_ERROR("no");
    return 0;
}
std::shared_ptr<cSimBodyLink> cSimCharacterGen::GetBodyPart(int idx) const
{
    return std::dynamic_pointer_cast<cSimBodyLink>(mLinkGenArray[idx]);
}

tVector cSimCharacterGen::GetBodyPartPos(int idx) const
{
    MIMIC_ERROR("no");
    return tVector::Zero();
}
tVector cSimCharacterGen::GetBodyPartVel(int idx) const
{
    return tVector::Zero();
}
std::shared_ptr<cSimBodyLink> cSimCharacterGen::GetRootPart() const
{
    return GetBodyPart(GetRootID());
}

void cSimCharacterGen::RegisterContacts(int contact_flags, int filter_flags)
{
    MIMIC_WARN("for generalized character, register contact doesn' work");
}
void cSimCharacterGen::UpdateContact(int contact_flags, int filter_flags)
{
    MIMIC_ERROR("no");
}
bool cSimCharacterGen::IsInContact() const
{
    MIMIC_ERROR("no");
    return false;
}
bool cSimCharacterGen::IsInContact(int idx) const
{
    MIMIC_ERROR("no");
    return false;
}
const tEigenArr<cContactManager::tContactPt> &
cSimCharacterGen::GetContactPts(int idx) const
{
    // return this->mContactHandle
    MIMIC_ERROR("no");
}
const tEigenArr<cContactManager::tContactPt> &
cSimCharacterGen::GetContactPts() const
{
    MIMIC_ERROR("no");
    return tEigenArr<cContactManager::tContactPt>();
}
const cContactManager::tContactHandle &
cSimCharacterGen::GetContactHandle() const
{
    MIMIC_ERROR("no");
    return cContactManager::tContactHandle();
}
const void cSimCharacterGen::GetTotalContactPts(Eigen::VectorXd &) const {}

/**
 * \brief               detect whether the character has falled to the ground
 */
bool cSimCharacterGen::HasFallen() const
{
    bool fallend = false;

    // if we enable the contact-ground fall (which means, if some specified body
    // parts are in contact with the ground, we will judge that the whole
    // character has falled to the ground)
    if (mEnableContactFall)
    {
        fallend |= CheckFallContact();
    }
    return false;
}
bool cSimCharacterGen::HasStumbled() const
{
    MIMIC_ERROR("no");
    return false;
}
bool cSimCharacterGen::HasVelExploded(double vel_threshold /* = 100.0*/) const
{
    return IsMaxVel();
}

bool cSimCharacterGen::IsValidBodyPart(int idx) const
{

    return idx >= 0 && idx < mLinkGenArray.size();
}
bool cSimCharacterGen::EnableBodyPartFallContact(int idx) const
{
    return mLinkGenArray[idx]->GetEnableFallContact();
}

/**
 * \brief               Set whether a body part can get in touch with the ground
 *          if a bodypart is set to "disable", then the character will be reset
 * when it is contacted with the ground
 */
void cSimCharacterGen::SetBodyPartFallContact(int idx, bool enable)
{
    mLinkGenArray[idx]->SetEnableFallContact(enable);
}

void cSimCharacterGen::SetController(std::shared_ptr<cCharController> ctrl)
{
    RemoveController();
    mController = ctrl;
}
void cSimCharacterGen::RemoveController()
{
    if (HasController())
    {
        mController.reset();
    }
}
bool cSimCharacterGen::HasController() const { return mController != nullptr; }
const std::shared_ptr<cCharController> &cSimCharacterGen::GetController()
{
    return mController;
}
const std::shared_ptr<cCharController> &cSimCharacterGen::GetController() const
{
    return mController;
}
void cSimCharacterGen::EnableController(bool enable)
{
    if (HasController())
    {
        mController->SetActive(enable);
    }
}

void cSimCharacterGen::ApplyForce(const tVector &force) {}
void cSimCharacterGen::ApplyForce(const tVector &force, const tVector &lo_pos)
{
}
void cSimCharacterGen::ApplyTorque(const tVector &torque) {}
void cSimCharacterGen::ClearForces() {}
void cSimCharacterGen::ApplyControlForces(const Eigen::VectorXd &tau) {}
void cSimCharacterGen::PlayPossum() {}

tVector cSimCharacterGen::GetPartColor(int part_id) const
{
    MIMIC_ERROR("no");
    return tVector::Zero();
}
double cSimCharacterGen::CalcTotalMass() const
{
    MIMIC_ERROR("no");
    return 0;
}

void cSimCharacterGen::SetLinearDamping(double damping) { MIMIC_ERROR("no"); }
void cSimCharacterGen::SetAngularDamping(double damping) { MIMIC_ERROR("no"); }

const std::shared_ptr<cWorldBase> &cSimCharacterGen::GetWorld() const
{
    return mWorld;
}

// cSimObj Interface
tVector cSimCharacterGen::GetPos() const
{
    MIMIC_ERROR("no");
    return tVector::Zero();
}
void cSimCharacterGen::SetPos(const tVector &pos) {}
void cSimCharacterGen::GetRotation(tVector &out_axis, double &out_theta) const
{
}
tQuaternion cSimCharacterGen ::GetRotation() const
{
    MIMIC_ERROR("no");
    return tQuaternion::Identity();
}
void cSimCharacterGen::SetRotation(const tVector &axis, double theta) {}
void cSimCharacterGen::SetRotation(const tQuaternion &q) {}
tMatrix cSimCharacterGen::GetWorldTransform() const
{
    MIMIC_ERROR("no");
    return tMatrix::Zero();
}

tVector cSimCharacterGen::GetLinearVelocity() const
{
    MIMIC_ERROR("no");
    return tVector::Zero();
}
tVector cSimCharacterGen::GetLinearVelocity(const tVector &local_pos) const
{
    MIMIC_ERROR("no");
    return tVector::Zero();
}
void cSimCharacterGen::SetLinearVelocity(const tVector &vel) {}
tVector cSimCharacterGen::GetAngularVelocity() const
{
    MIMIC_ERROR("no");
    return tVector::Zero();
}
void cSimCharacterGen::SetAngularVelocity(const tVector &vel) {}

short cSimCharacterGen::GetColGroup() const
{
    MIMIC_ERROR("no");
    return 0;
}
void cSimCharacterGen::SetColGroup(short col_group) {}
short cSimCharacterGen::GetColMask() const
{
    MIMIC_ERROR("no");
    return false;
}
void cSimCharacterGen::SetColMask(short col_mask) {}
void cSimCharacterGen::SetEnablejointTorqueControl(bool v_) {}
std::string cSimCharacterGen::GetCharFilename()
{
    MIMIC_ERROR("no");
    return "";
}

bool cSimCharacterGen::LoadBodyDefs(const std::string &char_file,
                                    Eigen::MatrixXd &out_body_defs)
{
    MIMIC_ERROR("no");
    return false;
}

bool cSimCharacterGen::BuildSimBody(const tParams &params)
{
    MIMIC_ERROR("no");
    return false;
}

/**
 * \brief               Build the body links (SimBodyLinkGen), put them all
 * together
 */
bool cSimCharacterGen::BuildBodyLinks()
{
    mLinkGenArray.clear();
    for (int i = 0; i < GetNumOfLinks(); i++)
    {
        auto link = std::shared_ptr<cSimBodyLinkGen>(new cSimBodyLinkGen());
        mLinkGenArray.push_back(link);
        // mLinkBaseArray.push_back(std::dynamic_pointer_cast<cSimBodyLink>(link));

        link->Init(this->mWorld, this, i);
    }
    return true;
}

btCollisionShape *
cSimCharacterGen::BuildCollisionShape(const cShape::eShape shape,
                                      const tVector &shape_size)
{
    return nullptr;
}

bool cSimCharacterGen::BuildJoints()
{
    mJointGenArray.clear();
    for (int i = 0; i < GetNumOfJoint(); i++)
    {
        auto joint = std::shared_ptr<cSimBodyJointGen>(new cSimBodyJointGen());
        joint->Init(mWorld, this, i);
        mJointGenArray.push_back(joint);
        // mJointBaseArray.push_back(
        //     std::dynamic_pointer_cast<cSimBodyJoint>(joint));
    }
    return true;
}
void cSimCharacterGen::BuildConsFactor(int joint_id, tVector &out_linear_factor,
                                       tVector &out_angular_factor) const
{
}
void cSimCharacterGen::BuildRootConsFactor(cKinTree::eJointType joint_type,
                                           tVector &out_linear_factor,
                                           tVector &out_angular_factor) const
{
}
bool cSimCharacterGen::FixedBase() const
{
    MIMIC_ERROR("no");
    return false;
}
void cSimCharacterGen::RemoveFromWorld() {}

void cSimCharacterGen::ClearJointTorques() {}
void cSimCharacterGen::UpdateJoints() {}
void cSimCharacterGen::UpdateLinkPos() {}
void cSimCharacterGen::UpdateLinkVel() {}

short cSimCharacterGen::GetPartColGroup(int part_id) const
{
    MIMIC_ERROR("no");
    return 0;
}
short cSimCharacterGen::GetPartColMask(int part_id) const
{
    MIMIC_ERROR("no");
    return 0;
}

void cSimCharacterGen::BuildJointPose(int joint_id,
                                      Eigen::VectorXd &out_pose) const
{
    MIMIC_ERROR("Lagragian characters don't build pose from joints");
}

/**
 * \brief               In lagragian character, we keep the status variable q
 * and qdot, never build pose and vel from joints
 */
void cSimCharacterGen::BuildJointVel(int joint_id,
                                     Eigen::VectorXd &out_vel) const
{
    MIMIC_ERROR("Lagragian characters don't build vel from joints");
}

/**
 * \brief       Build Pose from the the current character status
 */
void cSimCharacterGen::BuildPose(Eigen::VectorXd &out_pose) const
{
    out_pose = ConvertqToPose(mq);
}
void cSimCharacterGen::BuildVel(Eigen::VectorXd &out_vel) const
{
    out_vel = ConvertqdotToPoseVel(mqdot);
}

bool cSimCharacterGen::CheckFallContact() const
{
    // 角色: 检查摔倒接触
    int num_parts = GetNumBodyParts(); // 对于每一个接触
    for (int b = 0; b < num_parts; ++b)
    {
        if (IsValidBodyPart(b) && EnableBodyPartFallContact(b))
        {
            const auto &curr_part = GetBodyPart(b);
            bool has_contact = curr_part->IsInContact();
            if (has_contact)
            {
                return true;
            }
        }
    }
    return false;
}
const btCollisionObject *cSimCharacterGen::GetCollisionObject() const
{
    return nullptr;
}
btCollisionObject *cSimCharacterGen::GetCollisionObject() { return nullptr; }

eRotationOrder cSimCharacterGen::GetRotationOrder() const
{
    return eRotationOrder::XYZ;
}

/**
 * \brief               convert the lagragian q into pose
 *
 */
tVectorXd cSimCharacterGen::ConvertqToPose(const tVectorXd &q) const
{
    tVectorXd pose = tVectorXd::Zero(mPose0.size());
    int pose_st = 0, q_st = 0;

    // iteration on each joint
    // TODO: finish it here.
    int num_of_joints = GetNumOfJoint();
    for (int i = 0; i < num_of_joints; i++)
    {
        auto joint = static_cast<Joint *>(GetJointById(i));
        switch (joint->GetJointType())
        {
        case JointType::NONE_JOINT:
        {
            // for root
            pose.segment(pose_st, 3) = q.segment(q_st, 3);
            pose_st += 3, q_st += 3;

            tQuaternion root_rot = cMathUtil::EulerAnglesToQuaternion(
                cMathUtil::Expand(q.segment(q_st, 3), 0), eRotationOrder::XYZ);
            pose[pose_st] = root_rot.w();
            pose[pose_st + 1] = root_rot.x();
            pose[pose_st + 2] = root_rot.y();
            pose[pose_st + 3] = root_rot.z();
            pose_st += 4, q_st += 3;
        }
        break;
        case JointType::FIXED_JOINT:
            break;
        case JointType::REVOLUTE_JOINT:
        {
            pose[pose_st] = q[q_st];
            q_st++, pose_st++;
        }
        break;
        case JointType::SPHERICAL_JOINT:
        {
            tQuaternion joint_rot = cMathUtil::EulerAnglesToQuaternion(
                cMathUtil::Expand(q.segment(q_st, 3), 0), eRotationOrder::XYZ);
            pose[pose_st] = joint_rot.w();
            pose[pose_st + 1] = joint_rot.x();
            pose[pose_st + 2] = joint_rot.y();
            pose[pose_st + 3] = joint_rot.z();
            pose_st += 4, q_st += 3;
        }
        default:
            break;
        }
    }
    return pose;
}

tVectorXd cSimCharacterGen::ConvertqdotToPoseVel(const tVectorXd &qdot) const
{
    tVectorXd pose_vel = tVectorXd::Zero(mVel0.size());
    int pose_st = 0, q_st = 0;

    // iteration on each joint
    // TODO: finish it here.
    int num_of_joints = GetNumOfJoint();
    for (int i = 0; i < num_of_joints; i++)
    {
        auto joint = static_cast<Joint *>(GetJointById(i));
        switch (joint->GetJointType())
        {
        case JointType::NONE_JOINT:
        {
            // for root
            pose_vel.segment(pose_st, 3) = qdot.segment(q_st, 3);
            pose_st += 3, q_st += 3;

            pose_vel.segment(pose_st, 4) = ConvertEulerAngleVelToAxisAngleVel(
                cMathUtil::Expand(qdot.segment(q_st, 3), 0));
            pose_st += 4, q_st += 3;
        }
        break;
        case JointType::FIXED_JOINT:
            break;
        case JointType::REVOLUTE_JOINT:
        {
            pose_vel[pose_st] = qdot[q_st];
            q_st++, pose_st++;
        }
        break;
        case JointType::SPHERICAL_JOINT:
        {
            pose_vel.segment(pose_st, 4) = ConvertEulerAngleVelToAxisAngleVel(
                cMathUtil::Expand(qdot.segment(q_st, 3), 0));
            pose_st += 4, q_st += 3;
        }
        default:
            break;
        }
    }
    return pose_vel;
}

tVectorXd cSimCharacterGen::ConvertPoseToq(const tVectorXd &pose) const
{
    tVectorXd q = tVectorXd::Zero(mq.size()); // pose in generalized coordinate
    int pose_idx = 0, q_idx = 0;

    // iteration on each joint
    // TODO: finish it here.
    int num_of_joints = GetNumOfJoint();
    for (int i = 0; i < num_of_joints; i++)
    {
        auto joint = static_cast<Joint *>(GetJointById(i));
        switch (joint->GetJointType())
        {
        case JointType::NONE_JOINT:
        {
            // for root
            q.segment(q_idx, 3) = pose.segment(pose_idx, 3);
            pose_idx += 3, q_idx += 3;
            tVector w_x_y_z = pose.segment(pose_idx, 4);
            q.segment(q_idx, 3) =
                cMathUtil::QuaternionToEulerAngles(
                    tQuaternion(w_x_y_z[0], w_x_y_z[1], w_x_y_z[2], w_x_y_z[3]),
                    eRotationOrder::XYZ)
                    .segment(0, 3);
            pose_idx += 4, q_idx += 3;
        }
        break;
        case JointType::FIXED_JOINT:
            break;
        case JointType::REVOLUTE_JOINT:

        {
            q[q_idx] = pose[pose_idx];
            q_idx++, pose_idx++;
        }
        break;
        case JointType::SPHERICAL_JOINT:
        {
            tVector w_x_y_z = pose.segment(pose_idx, 4);
            q.segment(q_idx, 3) =
                cMathUtil::QuaternionToEulerAngles(
                    tQuaternion(w_x_y_z[0], w_x_y_z[1], w_x_y_z[2], w_x_y_z[3]),
                    eRotationOrder::XYZ)
                    .segment(0, 3);
            pose_idx += 4, q_idx += 3;
        }
        default:
            break;
        }
    }
    return q;
}
/**
 * \brief               Convert PoseVel to qdot in generalized coordinate
 * \param pose_vel      the pose velocity given by DeepMimic
 */
tVectorXd
cSimCharacterGen::ConvertPosevelToqdot(const tVectorXd &pose_vel) const
{
    /*
        1. assume dt = 0.01, axis angle *= dt
        2. axis angle converted to quaternion diff
        3. quaternion diff to euler angles
        4. euler angles /= dt
    */
    tVectorXd qdot =
        tVectorXd::Zero(mqdot.size()); // pose in generalized coordinate
    int pose_idx = 0, q_idx = 0;

    // iteration on each joint
    // TODO: finish it here.
    int num_of_joints = GetNumOfJoint();
    for (int i = 0; i < num_of_joints; i++)
    {

        auto joint = static_cast<Joint *>(GetJointById(i));
        switch (joint->GetJointType())
        {
        case JointType::NONE_JOINT:
        {
            // for root
            qdot.segment(q_idx, 3) = pose_vel.segment(pose_idx, 3);
            pose_idx += 3, q_idx += 3;
            qdot.segment(q_idx, 3) = ConvertAxisAngleVelToEulerAngleVel(
                                         pose_vel.segment(pose_idx, 4))
                                         .segment(0, 3);
            pose_idx += 4, q_idx += 3;
        }
        break;
        case JointType::FIXED_JOINT:
            break;
        case JointType::REVOLUTE_JOINT:

        {
            qdot[q_idx] = pose_vel[pose_idx];
            q_idx++, pose_idx++;
        }
        break;
        case JointType::SPHERICAL_JOINT:
        {
            qdot.segment(q_idx, 3) = ConvertAxisAngleVelToEulerAngleVel(
                                         pose_vel.segment(pose_idx, 4))
                                         .segment(0, 3);
            pose_idx += 4, q_idx += 3;
        }
        default:
            break;
        }
    }
    return qdot;
}

/**
 * \brief                   test convert methods
 */
void cSimCharacterGen::Test()
{
    // 1. set random value in q, then update the robotmodel dynamics
    // 2. set zero in qdot
    mq.setRandom();
    mqdot.setRandom();
    // mqdot.setZero();
    SetqAndqdot(mq, mqdot);

    // 3. save current links' position and orientation
    tEigenArr<tVector> pos_array(0);
    tEigenArr<tQuaternion> rot_array(0);
    for (auto link : mLinkGenArray)
    {
        pos_array.push_back(link->GetPos());
        rot_array.push_back(link->GetRotation());
    }

    // 4. load pose and posevel
    BuildPose(mPose);
    BuildVel(mVel);
    MIMIC_DEBUG("pose = {}", mPose.transpose());
    MIMIC_DEBUG("pose_vel = {}", mVel.transpose());

    tVectorXd q_new = ConvertPoseToq(mPose),
              qdot_new = ConvertPosevelToqdot(mVel);

    tVectorXd q_diff = (q_new - mq), qdot_diff = (qdot_new - mqdot);
    const double eps = 1e-6;
    MIMIC_DEBUG("link num {}", mLinkGenArray.size());
    if (q_diff.norm() < eps && qdot_diff.norm() < eps)
    {
        MIMIC_DEBUG("convert functions are verified succ");
    }
    else
    {
        MIMIC_DEBUG("q = {}", mq.transpose());
        MIMIC_DEBUG("qdot = {}", mqdot.transpose());
        MIMIC_DEBUG("q_new = {}", q_new.transpose());
        MIMIC_DEBUG("qdot_new = {}", qdot_new.transpose());
        MIMIC_DEBUG("q diff = {}\nqdot diff = {}", q_diff.transpose(),
                    qdot_diff.transpose());
        MIMIC_DEBUG("convert functions are verified failed");
    }

    exit(1);
}

/**
 * \brief           Init the default mPose, mPose0, mVel, mVel0 after loading
 * the ddefault state
 */
void cSimCharacterGen::InitDefaultState()
{
    // MIMIC_TRACE("begin to init default state for characters\n");
    // set up mPose0 and mVel0
    int total_size = 0; // time placeholder
    for (auto x : mJointGenArray)
    {
        total_size += x->GetPoseSize();
    }
    mPose0.resize(total_size);
    mPose0.setZero();
    mVel0.resize(total_size);
    mVel0.setZero();
    mPose = mPose0;
    mVel = mVel0;
    SetPose(mPose);
    SetVel(mVel);
    // MIMIC_WARN("cSimCharacterGen::InitDefaultState need to set the pose &
    // vel");

    MIMIC_TRACE("Character InitDefaultState done");
}

/**
 * \brief                   Set the joint position and velocity of this
 * character Note that the mPose and mVel should be maintained as well
 */
void cSimCharacterGen::SetqAndqdot(const tVectorXd &q, const tVectorXd &qdot)
{
    cRobotModelDynamics::SetqAndqdot(q, qdot);
    BuildPose(mPose);
    BuildVel(mVel);
}

/**
 * \brief                   Set the velocity of this character
 * Note that the mVel are also changed simultaneously
 */
void cSimCharacterGen::Setqdot(const tVectorXd &qdot)
{
    cRobotModelDynamics::Setqdot(qdot);
    BuildVel(mVel);
}

// ------------------------- SimCharacter
void cSimCharacterGen::RotateRoot(const tQuaternion &rot)
{
    tQuaternion new_root_rot = rot * GetRootRotation();
    new_root_rot.normalize();
    SetRootRotation(new_root_rot);
}

/**
 * \brief
 */
double cSimCharacterGen::CalcHeading() const
{
    tVector ref_dir = tVector(1, 0, 0, 0);
    tVector rot_dir = cMathUtil::QuatRotVec(GetRootRotation(), ref_dir);
    double theta = std::atan2(-rot_dir[2], rot_dir[0]);
    return theta;
}

/**
 * \brief               Equaivlant with cKinTree::BuildOriginTrans
 *
 * Calculate the transformation matrix, which can move the skeleton to be origin
 * point, also make its orientation is parallel with (0, 0, 1)
 */
tMatrix cSimCharacterGen::BuildOriginTrans() const
{
    tMatrix trans_mat = cMathUtil::TranslateMat(-GetRootPos());
    tMatrix rot_mat = cMathUtil::RotMat(CalcHeadingRot());
    return rot_mat * trans_mat;
}

/**
 * \brief               Judge whether this joint "joint_id" is an end-effector
 */
bool cSimCharacterGen::IsEndEffector(int joint_id) const
{
    MIMIC_ASSERT(joint_id >= 0 && joint_id <= mLinkGenArray.size());
    return 0 == GetLinkById(joint_id)->GetNumOfChildren();
}

/**
 * \brief               Calculate the length of this chain from root to
 * "joint_id"
 * \param joint_id
 */
double cSimCharacterGen::CalcJointChainLength(int joint_id) const
{
    // 获取每一个joint，在其parent坐标系下的位移的范数，加在一起。
    // 如果是endeffector，就还要加上一个link的最大长度
    MIMIC_ASSERT(joint_id >= 0 && joint_id < mLinkGenArray.size());
    double length = mLinkGenArray[joint_id]->GetLinkMaxLength();
    while (joint_id != -1)
    {

        length += GetJointById(joint_id)->GetLocalPos().norm();
        if (joint_id == GetRootID())
            joint_id = -1;
    }
    return length;
}

int cSimCharacterGen::CalcNumEndEffectors() const
{
    int num = 0;
    for (auto x : mLinkGenArray)
    {
        if (x->IsEndEffector())
            num++;
    }
    return num;
}

/**
 * \brief                       Fetch the weight w.r.t this joint
 * used in the calculation of the mimic "reward". each joint leverages a
 * different weight
 */
double cSimCharacterGen::GetJointDiffWeight(int joint_id) const
{
    return mJointGenArray[joint_id]->GetJointDiffWeight();
}

bool cSimCharacterGen::HasDrawShapes() const { return true; }
const Eigen::MatrixXd &cSimCharacterGen::GetDrawShapeDefs() const
{
    // MIMIC_ERROR("shouldn't be called");
    // return Eigen::MatrixXd::Zero(0, 0);
    return mDrawShapeDefs;
}
const std::shared_ptr<cDrawMesh> &cSimCharacterGen::GetMesh(int i) const
{
    MIMIC_ERROR("shouldn't be called");
    return std::shared_ptr<cDrawMesh>();
}
int cSimCharacterGen::GetNumMeshes() const
{
    MIMIC_ERROR("shouldn't be called");
    return 0;
}
std::string cSimCharacterGen::GetBodyName(int id) const
{
    MIMIC_ERROR("shouldn't be called");
    return "";
}
std::string cSimCharacterGen::GetJointName(int id) const
{
    MIMIC_ERROR("shouldn't be called");
    return "";
}
std::string cSimCharacterGen::GetDrawShapeName(int id) const
{
    MIMIC_ERROR("shouldn't be called");
    return "";
}

int cSimCharacterGen::GetNumJoints() const { return mJointGenArray.size(); }