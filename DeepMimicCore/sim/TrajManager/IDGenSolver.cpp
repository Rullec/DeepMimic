#include "BulletGenDynamics/btGenModel/Link.h"
#include "BulletGenDynamics/btGenSolver/ContactSolver.h"
#include "IDSolver.h"
#include "sim/SimItems/SimCharacterGen.h"
#include "sim/World/GenWorld.h"
#include <iostream>

/**
 * Inverse Dynamics implementaion for genralized coordinate backend
 * "cSimCharacterGen"
 */
cSimCharacterGen *DownCastGen(cSimCharacterBase *sim_char)
{
    return dynamic_cast<cSimCharacterGen *>(sim_char);
}
const cSimCharacterGen *DownCastGen(const cSimCharacterBase *sim_char)
{
    return dynamic_cast<const cSimCharacterGen *>(sim_char);
}
void cIDSolver::InitGenVariables()
{
    if (mSimCharType == eSimCharacterType::Generalized)
    {
        mDof = mSimChar->GetNumDof();
        mNumLinks = mSimChar->GetNumBodyParts();
        // std::dynamic_pointer_cast<cGenWorld>(mWorld)->GetInternalWorld
        // mGenWorld =dynamic_cast<btGenWorld *> mWorld->
        // std::cout << "mDof = " << mDof << std::endl;
        // std::cout << "mnumlinks = " << mNumLinks << std::endl;
        // MIMIC_ERROR("Done to intiialize the generaalized variables");
    }
}

void cIDSolver::RecordMultibodyInfoGen(const cSimCharacterBase *sim_char,
                                       tEigenArr<tMatrix> &local_to_world_rot,
                                       tEigenArr<tVector> &link_pos_world,
                                       tEigenArr<tVector> &link_omega_world,
                                       tEigenArr<tVector> &link_vel_world)
{
    MIMIC_ASSERT(sim_char->GetCharType() == eSimCharacterType::Generalized);
    int num_of_links = sim_char->GetNumBodyParts();
    local_to_world_rot.resize(num_of_links);
    link_pos_world.resize(num_of_links);
    link_omega_world.resize(num_of_links);
    link_vel_world.resize(num_of_links);
    auto gen_char = dynamic_cast<const cSimCharacterGen *>(sim_char);
    for (int i = 0; i < num_of_links; i++)
    {
        auto link = dynamic_cast<Link *>(gen_char->GetLinkById(i));
        link_omega_world[i] = cMathUtil::Expand(link->GetLinkOmega(), 0);
        link_vel_world[i] = cMathUtil::Expand(link->GetLinkVel(), 0);
        local_to_world_rot[i] =
            cMathUtil::RotMat(sim_char->GetBodyPart(i)->GetRotation());
        link_pos_world[i] = sim_char->GetBodyPartPos(i);
    }
}

void cIDSolver::RecordMultibodyInfoGen(cSimCharacterBase *sim_char,
                                       tEigenArr<tMatrix> &local_to_world_rot,
                                       tEigenArr<tVector> &link_pos_world)
{
    MIMIC_ASSERT(sim_char->GetCharType() == eSimCharacterType::Generalized);
    int num_of_links = sim_char->GetNumBodyParts();
    local_to_world_rot.resize(num_of_links);
    link_pos_world.resize(num_of_links);
    for (int i = 0; i < num_of_links; i++)
    {
        local_to_world_rot[i] =
            cMathUtil::RotMat(sim_char->GetBodyPart(i)->GetRotation());
        link_pos_world[i] = sim_char->GetBodyPartPos(i);
    }
}
void cIDSolver::RecordGeneralizedInfoGen(cSimCharacterBase *sim_char,
                                         tVectorXd &q, tVectorXd &q_dot)
{
    MIMIC_ASSERT(sim_char->GetCharType() == eSimCharacterType::Generalized);
    auto gen_char = DownCastGen(sim_char);
    q = gen_char->Getq();
    q_dot = gen_char->Getqdot();
}

void cIDSolver::SetGeneralizedPosGen(cSimCharacterBase *sim_char,
                                     const tVectorXd &q)
{
    MIMIC_ASSERT(sim_char->GetCharType() == eSimCharacterType::Generalized);
    auto gen_char = DownCastGen(sim_char);
    gen_char->SetqAndqdot(q, gen_char->Getqdot());
}

/**
 * \brief           Record the joint forces for the genralized character
 *
 *
 * Current the character is controlled by generalize force on all
 * non-root freedoms, so the joint forces recorded are only literally slices of
 * the whole gen force vector on their own freedoms
 * 1. for root joint we cannot apply actuated torque on that, ignore
 * 2. for revolute joint, the torque recorded is a scalar, tVector(x, 0, 0, 0)
 * 3. for spherical joint, the torque recorded is a 3d-vector, tVector(x, y, z,
 * 0)
 * 4. for fixed joint, the torque is fully zero vector
 */
void cIDSolver::RecordJointForcesGen(tEigenArr<tVector> &mJointForces) const
{
    MIMIC_ASSERT(this->mSimCharType == eSimCharacterType::Generalized);

    // non-root jont cannot be actuated
    mJointForces.resize(mNumLinks - 1);
    // auto gen_char = Down
    auto gen_char = DownCastGen(mSimChar);
    tVectorXd gen_force = gen_char->GetGeneralizedForce();
    int offset = gen_char->GetJointById(0)->GetNumOfFreedom();
    // MIMIC_INFO("current gen force {}", gen_force.transpose());

    for (int joint_id = 1; joint_id < mNumLinks; joint_id++)
    {
        auto joint = gen_char->GetJointById(joint_id);
        int dof = joint->GetNumOfFreedom();
        mJointForces[joint_id - 1].setZero();
        switch (joint->GetJointType())
        {
        case JointType::REVOLUTE_JOINT:
            mJointForces[joint_id - 1].segment(0, dof) =
                gen_force.segment(offset, dof);
            break;
        case JointType::SPHERICAL_JOINT:
            mJointForces[joint_id - 1].segment(0, dof) =
                gen_force.segment(offset, dof);
            break;
        case JointType::FIXED_JOINT:
            mJointForces[joint_id - 1].setZero();
            break;
        default:
            MIMIC_ERROR("unsupported joint type");
            break;
        }
        // MIMIC_INFO("joint {} force {}", joint_id,
        //            mJointForces[joint_id - 1].transpose());
        offset += dof;
    }
}

/**
 * \brief               Record contact forces of genrealize character
 */
void cIDSolver::RecordContactForcesGen(
    tEigenArr<tContactForceInfo> &mContactForces, double mCurTimestep) const
{
    mContactForces.clear();
    const auto &contact_force_info =
        dynamic_cast<cGenWorld *>(mWorld)->GetContactInfo();
    tContactForceInfo info;
    for (auto &f : contact_force_info)
    {
        if (f->mObj->GetType() == eColObjType::RobotCollder)
        {
            info.mForce = f->mForce;
            info.mId = dynamic_cast<btGenRobotCollider *>(f->mObj)->mLinkId;
            info.mPos = f->mWorldPos;
            info.mIsSelfCollision = f->mIsSelfCollision;
            mContactForces.push_back(info);
            // MIMIC_INFO("for link {} pos {} force {} is_self_collision {}",
            //            info.mId, info.mPos.transpose(),
            //            info.mForce.transpose(), info.mIsSelfCollision);
        }
    };
    // MIMIC_INFO("total contact force num {}", mContactForces.size());
    // MIMIC_ERROR("hasn't been implemented");
}

/**
 * \brief               Solve the control torque by given the generalized info
 * (q, qdot, qddot) and all external forces
 *  1. save the current status of char
 *  2. set the q and qdot
 *  3. calculate the total Q = Mqddot + Cqdot
 *  4. Q_ctrl = Q minus the gravity, the external force and the contact forces
 *  5. shape the output "solved_joint_forces"
 *  6. restore the status of char previously
 */
void cIDSolver::SolveIDSingleStepGen(
    tEigenArr<tVector> &solved_joint_forces,
    const tEigenArr<tContactForceInfo> &contact_forces,
    const tEigenArr<tVector> &link_pos, const tEigenArr<tMatrix> &link_rot,
    const tVectorXd &q, const tVectorXd &qdot, const tVectorXd &qddot,
    int frame_id, const tEigenArr<tVector> &mExternalForces,
    const tEigenArr<tVector> &mExternalTorques) const
{

    // 1. save the cur status of char
    auto gen_char = dynamic_cast<cRobotModelDynamics *>(mSimChar);
    // std::cout << "[id] before q = " << gen_char->Getq().transpose()
    //           << std::endl;
    gen_char->PushState("solve_id");

    // 2. set the q and qdot
    int dof = gen_char->GetNumOfFreedom();
    MIMIC_ASSERT(q.size() == dof);
    MIMIC_ASSERT(qdot.size() == dof);
    MIMIC_ASSERT(qddot.size() == dof);
    gen_char->SetqAndqdot(q, qdot);
    gen_char->ClearForce();

    // 3. calculate the total Q
    tVectorXd Q_total =
        gen_char->GetMassMatrix() * qddot +
        (gen_char->GetCoriolisMatrix() + gen_char->GetDampingMatrix()) * qdot;

    // 4. get the Q ctrl
    tVectorXd Q_ctrl = Q_total;
    // 4.1 minus the gravity
    tVector gravity = mWorld->GetGravity();
    Q_ctrl -= gen_char->CalcGenGravity(gravity);

    // 4.2 minus the contact force
    tMatrixXd jac;
    for (auto f : contact_forces)
    {
        gen_char->ComputeJacobiByGivenPointTotalDOFWorldFrame(
            f.mId, f.mPos.segment(0, 3), jac);
        Q_ctrl -= jac.transpose() * f.mForce.segment(0, 3);
    }

    // 4.3 minus the external forces
    MIMIC_ASSERT(mExternalForces.size() == mNumLinks);
    MIMIC_ASSERT(mExternalTorques.size() == mNumLinks);

    for (int id = 0; id < mNumLinks; id++)
    {
        auto link = static_cast<Link *>(gen_char->GetLinkById(id));
        Q_ctrl -=
            link->GetJKv().transpose() * mExternalForces[id].segment(0, 3);
        Q_ctrl -=
            link->GetJKw().transpose() * mExternalTorques[id].segment(0, 3);
    }

    // 5. shape the final control forces
    MIMIC_ASSERT(solved_joint_forces.size() == (mNumLinks - 1));
    int offset = gen_char->GetJointById(0)->GetNumOfFreedom();
    for (int id = 1; id < mNumLinks; id++)
    {
        auto joint = gen_char->GetJointById(id);
        int dof = joint->GetNumOfFreedom();
        solved_joint_forces[id - 1].setZero();

        solved_joint_forces[id - 1].segment(0, dof) =
            Q_ctrl.segment(offset, dof);
        // std::cout << "[solved] joint " << id - 1
        //           << " force = " << solved_joint_forces[id].transpose()
        //           << std::endl;
        offset += dof;
    }

    // 6. restore the status
    gen_char->PopState("solve_id");
    // std::cout << "[id] after q = " << gen_char->Getq().transpose() <<
    // std::endl;
}

void cIDSolver::SetGeneralizedVelGen(const tVectorXd &qdot)
{
    DownCastGen(mSimChar)->Setqdot(qdot);
}

/**
 * \brief               Calculate qdot by qcur and qbefore
 */
tVectorXd cIDSolver::CalcGeneralizedVelGen(const tVectorXd &q_before,
                                           const tVectorXd &q_after,
                                           double timestep) const
{
    return (q_after - q_before) / timestep;
}