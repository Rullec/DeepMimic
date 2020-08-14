#include "ImpPDGenController.h"
#include "sim/SimItems/SimCharacterGen.h"
#include "util/LogUtil.hpp"
#include <iostream>

typedef cImpPDGenController cPDCtrl;

cPDCtrl::cImpPDGenController()
{
    MIMIC_INFO("cImpPDGenController constructed");
    Clear();
}
cPDCtrl::~cImpPDGenController() {}
/**
 * \brief           Given Kp and Kd parameters, init this stable PD controller
 */
void cPDCtrl::Init(cSimCharacterGen *gen_char, const tVectorXd &kp,
                   const tVectorXd &kd)
{
    mChar = gen_char;
    InitGains(kp, kd);
    VerifyController();
}

void cPDCtrl::Clear()
{
    mKp.resize(0);
    mKd.resize(0);
    mTarget_q.resize(0);
    mTarget_qdot.resize(0);
    mChar = nullptr;
}

void cPDCtrl::SetPDTarget(const tVectorXd &q, const tVectorXd &qdot)
{
    MIMIC_ASSERT(q.size() == GetPDTargetSize());
    MIMIC_ASSERT(qdot.size() == GetPDTargetSize());
    mTarget_q = q;
    mTarget_qdot = qdot;
}

/**
 * \brief           Verify the PD coefficient, make sure that their size is
 * correct, their value are non-negative
 */
void cPDCtrl::VerifyController()
{
    MIMIC_INFO("begin to verify cImpPDGenController");

    int target_size = GetPDTargetSize();

    if (mKp.size() != target_size || mKd.size() != target_size)
    {
        MIMIC_ERROR("cImpPDGenController Kp/Kd invalid size {}/{} != {}",
                    mKp.size(), mKd.size(), target_size);
    }

    MIMIC_ASSERT(mKp.minCoeff() >= 0 && mKd.minCoeff() >= 0);
}

/**
 * \brief                   The size of given PD target should be the same as q
 * without root freedom
 */
int cPDCtrl::GetPDTargetSize()
{
    MIMIC_ASSERT(mChar != nullptr);

    int size = mChar->GetNumOfFreedom();
    // int size = mChar->GetNumOfFreedom() -
    // mChar->GetRoot()->GetNumOfFreedom();
    // MIMIC_INFO("PD target size {}", size);
    return size;
}

/**
 * \brief           Calculate the control force
 */
void cPDCtrl::UpdateControlForce(double dt, tVectorXd &out_tau)
{
    // 1. check the PD target "q" and "qdot" should the same as the dof of this
    // character. they are in generalized coordinate
    MIMIC_ASSERT(mTarget_qdot.size() == GetPDTargetSize() &&
                 mTarget_q.size() == GetPDTargetSize() &&
                 "the PD target has been set up well");
    CheckVelExplode();

    tVectorXd q = mChar->Getq(), qdot = mChar->Getqdot(),
              qddot = mChar->Getqddot();

    out_tau = -mKp.cwiseProduct(q + dt * qdot - mTarget_q) -
              mKd.cwiseProduct(qdot + dt * qddot - mTarget_qdot);

    // for underactutated system, the first 6 root doms should be ignored now
    PostProcessControlForce(out_tau);
    // MIMIC_INFO("out tau = {}", out_tau.transpose());
}

/**
 * \brief               Extract & init the mKp and mKd used in the computation
 * by given a joint-size-long kp & kd
 */
void cPDCtrl::InitGains(const tVectorXd &kp, const tVectorXd &kd)
{
    // 1. judge input
    int num_of_joints = mChar->GetNumOfJoint(); // except roots
    MIMIC_ASSERT(kp.size() == num_of_joints && kd.size() == num_of_joints);

    // 2. expand the joint-size-long kp & kd -> dof-long kp & kd
    int num_of_freedom = mChar->GetNumOfFreedom();
    mKp = tVectorXd::Zero(num_of_freedom);
    mKd = tVectorXd::Zero(num_of_freedom);

    int st_pos = 0;
    for (int i = 0; i < num_of_joints; i++)
    {
        double joint_kp = kp[i], joint_kd = kd[i];
        auto joint = mChar->GetJointById(i);
        int joint_dof = joint->GetNumOfFreedom();

        mKp.segment(st_pos, joint_dof).fill(joint_kp);
        mKd.segment(st_pos, joint_dof).fill(joint_kd);
        st_pos += joint_dof;
    }
    MIMIC_INFO("Init gains: kp = {}, kd = {}", mKp.transpose(),
               mKd.transpose());
}

/**
 * \brief                   Check and warn the velocity explode in qdot
 * (generalized velocity) when we try to calculate the control forces
 */
void cPDCtrl::CheckVelExplode()
{
    tVectorXd qdot = mChar->Getqdot();
    for (int i = 0; i < mChar->GetNumOfFreedom(); i++)
    {
        if ((std::fabs(qdot[i])) >= mChar->GetMaxVel() - 1e-5)
        {
            MIMIC_WARN(
                "qdot {} = {} exceed the max_vel {}, belongs to joint {}", i,
                qdot[i], mChar->GetMaxVel(),
                mChar->GetJointByDofId(i)->GetName());
        }
    }
}

void cPDCtrl::PostProcessControlForce(tVectorXd &out_tau)
{
    auto root_joint = mChar->GetRoot();
    if (root_joint->GetJointType() == JointType::NONE_JOINT)
    {
        // MIMIC_INFO("set root joint(dof {}) tau to zero",
        //            root_joint->GetNumOfFreedom());
        out_tau.segment(0, root_joint->GetNumOfFreedom()).setZero();
    }
}